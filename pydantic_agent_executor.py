#!/usr/bin/env python3
"""
PydanticAI-based ReAct Agent for Natural Gas Pipeline Data Analysis

This agent uses the ReAct (Reasoning-Acting-Observing) framework to answer
complex questions about natural gas pipeline transportation data.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import tiktoken
from retry_ledger import RetryLedger, set_current_ledger, log_execution, log_pre_dispatch, log_post_process, print_retry_summary
import yaml
from dotenv import load_dotenv
from jinja2 import Template
from pydantic import BaseModel, Field, ConfigDict
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel

# Load environment variables
load_dotenv()


class ToolTiming(BaseModel):
    """Track timing information for individual tool calls."""
    tool_name: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error: Optional[str] = None


class WriteRunArgs(BaseModel):
    """Arguments for write_file_and_run_python tool with aliases for common parameter name variations."""
    file_path: str = Field(..., description="Relative path to the Python file to write and execute", alias="filename")
    content: str = Field(..., description="Python code content to write to the file")

    model_config = ConfigDict(populate_by_name=True)  # Accept both file_path and filename


class WriteRunResult(BaseModel):
    """Structured result for write_file_and_run_python tool execution."""
    file_path: str
    exit_code: int
    stdout_log: str
    stderr_log: Optional[str] = None
    duration_s: float
    success: bool
    stdout_tail: Optional[str] = None
    stderr_tail: Optional[str] = None
    task_id: str
    content_length: int

    def __len__(self) -> int:
        """Return length of stdout_log for compatibility with len() calls."""
        return len(self.stdout_log) if self.stdout_log else 0

    def __str__(self) -> str:
        """Return string representation compatible with tool result expectations."""
        if self.success:
            return f"Successfully executed {self.file_path} (exit code: {self.exit_code}, duration: {self.duration_s:.1f}s)"
        else:
            return f"Failed to execute {self.file_path} (exit code: {self.exit_code})"


class AgentState(BaseModel):
    """State tracking for the agent run."""
    workspace_dir: Path
    query: str
    query_id: str
    dataset_description: str
    analysis_type: str
    rubric: Dict[str, Any]
    console_updates_enabled: bool = True
    tool_timings: List[ToolTiming] = []
    total_start_time: float
    react_log_path: Optional[Path] = None
    cost_info: Optional[Dict[str, Any]] = None

    def add_tool_timing(self, tool_name: str, start_time: float, end_time: float,
                       success: bool, error: Optional[str] = None):
        """Add timing information for a tool call."""
        timing = ToolTiming(
            tool_name=tool_name,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            success=success,
            error=error
        )
        self.tool_timings.append(timing)

    def log_react_event(self, event_type: str, data: Dict[str, Any]):
        """Log a ReAct event to the react_log.jsonl file."""
        if self.react_log_path:
            try:
                # Ensure the parent directory exists
                self.react_log_path.parent.mkdir(parents=True, exist_ok=True)

                event = {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": event_type,
                    "query_id": self.query_id,
                    **data
                }
                with open(self.react_log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(event, default=str) + '\n')
            except Exception as e:
                print(f"Warning: Could not write to react log: {e}")

    def get_timing_summary(self) -> Dict[str, Any]:
        """Get comprehensive timing summary."""
        total_elapsed = time.time() - self.total_start_time
        tool_time = sum(t.duration for t in self.tool_timings)

        timing_by_tool = {}
        for timing in self.tool_timings:
            if timing.tool_name not in timing_by_tool:
                timing_by_tool[timing.tool_name] = {
                    "count": 0,
                    "total_duration": 0,
                    "successful_calls": 0,
                    "failed_calls": 0
                }

            timing_by_tool[timing.tool_name]["count"] += 1
            timing_by_tool[timing.tool_name]["total_duration"] += timing.duration
            if timing.success:
                timing_by_tool[timing.tool_name]["successful_calls"] += 1
            else:
                timing_by_tool[timing.tool_name]["failed_calls"] += 1

        return {
            "total_elapsed_time": total_elapsed,
            "total_tool_time": tool_time,
            "thinking_time": total_elapsed - tool_time,
            "tool_efficiency": tool_time / total_elapsed if total_elapsed > 0 else 0,
            "tool_breakdown": timing_by_tool,
            "individual_tool_calls": [t.model_dump() for t in self.tool_timings]
        }


def load_pricing_data(model_name: str) -> Dict[str, Dict[str, float]]:
    """Load pricing data from appropriate TSV file based on model type."""
    if model_name.startswith("openai:"):
        pricing_file = Path("config/openai_pricing.tsv")
    else:
        pricing_file = Path("config/anthropic_pricing.tsv")

    if not pricing_file.exists():
        raise FileNotFoundError(f"Pricing file not found: {pricing_file}")

    import pandas as pd
    df = pd.read_csv(pricing_file, sep='\t')

    pricing = {}
    for _, row in df.iterrows():
        model_name_lower = row['Model'].lower()
        try:
            input_str = str(row['Input']).replace('$', '')
            output_str = str(row['Output']).replace('$', '')

            # Skip rows with missing data (represented as '-')
            if input_str == '-' or output_str == '-':
                continue

            input_price = float(input_str)
            output_price = float(output_str)
            pricing[model_name_lower] = {
                "input_per_1m": input_price,
                "output_per_1m": output_price
            }
        except (ValueError, KeyError):
            # Skip rows with invalid data
            continue

    if not pricing:
        raise ValueError(f"No valid pricing data found in {pricing_file}")

    return pricing


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens using tiktoken for accurate measurement."""
    try:
        # Map model names to tiktoken encodings
        encoding_map = {
            "gpt-4o-mini": "o200k_base",
            "gpt-4o": "o200k_base",
            "gpt-4": "cl100k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "claude-sonnet-4-20250514": "o200k_base",  # Use OpenAI encoding for Claude
            "claude-3-5-haiku-20241022": "o200k_base"
        }

        clean_model = model.replace("anthropic:", "").replace("openai:", "")
        encoding_name = encoding_map.get(clean_model, "o200k_base")
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to rough estimation
        return len(text) // 4


def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> Dict[str, Any]:
    """Calculate the cost based on token usage and model pricing."""
    # Local models (Ollama) are free
    if model.startswith("ollama:"):
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
            "model": model
        }

    clean_model = model.replace("anthropic:", "").replace("openai:", "")
    pricing_data = load_pricing_data(model)

    if clean_model not in pricing_data:
        # Try to find a match by checking if the model name contains the key
        found_model = None
        for pricing_model in pricing_data.keys():
            if pricing_model in clean_model.lower():
                found_model = pricing_model
                break

        if not found_model:
            raise ValueError(f"Pricing not found for model: {clean_model}")
        clean_model = found_model

    model_pricing = pricing_data[clean_model]
    input_cost = input_tokens * model_pricing["input_per_1m"] / 1_000_000
    output_cost = output_tokens * model_pricing["output_per_1m"] / 1_000_000

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost
    }


def console_update(ctx: RunContext[AgentState], message: str) -> str:
    """Print console update message if enabled."""
    state = ctx.deps
    if state.console_updates_enabled:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    return message


def write_file(ctx: RunContext[AgentState], file_path: str, content: str) -> str:
    """Write content to a file in the workspace."""
    state = ctx.deps
    start_time = time.time()

    # Log tool call start
    state.log_react_event("tool_call_start", {
        "tool_name": "write_file",
        "args": {"file_path": file_path, "content_length": len(content)}
    })

    try:
        full_path = state.workspace_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

        end_time = time.time()
        state.add_tool_timing("write_file", start_time, end_time, True)

        result = f"Successfully wrote {len(content)} characters to {file_path}"

        # Log tool call completion
        state.log_react_event("tool_call_complete", {
            "tool_name": "write_file",
            "success": True,
            "result": result,
            "duration": end_time - start_time
        })

        return result

    except Exception as e:
        end_time = time.time()
        error_msg = f"Error writing file {file_path}: {str(e)}"
        state.add_tool_timing("write_file", start_time, end_time, False, error_msg)

        # Log tool call completion with error
        state.log_react_event("tool_call_complete", {
            "tool_name": "write_file",
            "success": False,
            "error": error_msg,
            "duration": end_time - start_time
        })

        return error_msg


def run_python(ctx: RunContext[AgentState], script_path: str) -> str:
    """Execute a Python script in the workspace."""
    state = ctx.deps
    start_time = time.time()

    # Log tool call start
    state.log_react_event("tool_call_start", {
        "tool_name": "run_python",
        "args": {"script_path": script_path}
    })

    try:
        full_path = state.workspace_dir / script_path
        if not full_path.exists():
            end_time = time.time()
            error_msg = f"Script not found: {script_path}"
            state.add_tool_timing("run_python", start_time, end_time, False, error_msg)
            return error_msg

        # Create logs directory in parent workspace
        logs_dir = state.workspace_dir.parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique task ID for this execution
        task_id = f"{int(time.time())}-{script_path.replace('/', '_').replace('.py', '')}"
        stdout_log = (logs_dir / f"stdout-{task_id}.log").resolve()
        stderr_log = (logs_dir / f"stderr-{task_id}.log").resolve()

        # Change to workspace directory for execution
        original_cwd = os.getcwd()
        os.chdir(state.workspace_dir)

        try:
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            # Write stdout and stderr to separate log files
            try:
                if result.stdout:
                    with open(stdout_log, 'w', encoding='utf-8') as f:
                        f.write(result.stdout)
                if result.stderr:
                    with open(stderr_log, 'w', encoding='utf-8') as f:
                        f.write(result.stderr)
            except IOError as log_error:
                # Log file creation failed, but continue execution
                print(f"Warning: Could not write log files: {log_error}")
                pass

            output = f"Exit code: {result.returncode}\n"
            output += f"Logs written to: stdout-{task_id}.log, stderr-{task_id}.log\n"
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"

            end_time = time.time()
            success = result.returncode == 0

            # Log tool call completion
            state.log_react_event("tool_call_complete", {
                "tool_name": "run_python",
                "success": success,
                "exit_code": result.returncode,
                "task_id": task_id,
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "duration": end_time - start_time
            })

            if not success:
                error_msg = f"Script failed with exit code {result.returncode}"
                state.add_tool_timing("run_python", start_time, end_time, False, error_msg)
            else:
                state.add_tool_timing("run_python", start_time, end_time, True)

            return output

        finally:
            os.chdir(original_cwd)

    except subprocess.TimeoutExpired:
        end_time = time.time()
        error_msg = "Script execution timed out (5 minutes)"
        state.add_tool_timing("run_python", start_time, end_time, False, error_msg)
        return error_msg

    except Exception as e:
        end_time = time.time()
        error_msg = f"Error executing script: {str(e)}"
        state.add_tool_timing("run_python", start_time, end_time, False, error_msg)
        return error_msg


def read_file(ctx: RunContext[AgentState], file_path: str) -> str:
    """Read content from a file in the workspace."""
    state = ctx.deps
    start_time = time.time()

    try:
        full_path = state.workspace_dir / file_path
        if not full_path.exists():
            end_time = time.time()
            error_msg = f"File not found: {file_path}"
            state.add_tool_timing("read_file", start_time, end_time, False, error_msg)
            return error_msg

        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()

        end_time = time.time()
        state.add_tool_timing("read_file", start_time, end_time, True)

        return content

    except Exception as e:
        end_time = time.time()
        error_msg = f"Error reading file {file_path}: {str(e)}"
        state.add_tool_timing("read_file", start_time, end_time, False, error_msg)
        return error_msg


def list_files(ctx: RunContext[AgentState], directory: str = ".") -> str:
    """List files in a directory within the workspace."""
    state = ctx.deps
    start_time = time.time()

    try:
        full_path = state.workspace_dir / directory
        if not full_path.exists():
            end_time = time.time()
            error_msg = f"Directory not found: {directory}"
            state.add_tool_timing("list_files", start_time, end_time, False, error_msg)
            return error_msg

        if not full_path.is_dir():
            end_time = time.time()
            error_msg = f"Path is not a directory: {directory}"
            state.add_tool_timing("list_files", start_time, end_time, False, error_msg)
            return error_msg

        files = []
        for item in full_path.iterdir():
            if item.is_file():
                files.append(f"📄 {item.name}")
            elif item.is_dir():
                files.append(f"📁 {item.name}/")

        end_time = time.time()
        state.add_tool_timing("list_files", start_time, end_time, True)

        if not files:
            return f"Directory {directory} is empty"

        return f"Contents of {directory}:\n" + "\n".join(sorted(files))

    except Exception as e:
        end_time = time.time()
        error_msg = f"Error listing directory {directory}: {str(e)}"
        state.add_tool_timing("list_files", start_time, end_time, False, error_msg)
        return error_msg


def write_file_and_run_python(ctx: RunContext[AgentState], args: WriteRunArgs) -> WriteRunResult:
    """Write a Python script file and immediately execute it. This fused operation reduces tool calls and avoids rewrites.

    Args:
        args: WriteRunArgs containing file_path (or filename) and content

    Returns:
        WriteRunResult with structured execution details
    """
    state = ctx.deps
    start_time = time.time()

    # PARAMETER VALIDATION: Ensure both required parameters are provided
    if not hasattr(args, 'file_path') or not args.file_path:
        raise ValueError("🚨 MISSING PARAMETER: file_path is required for write_file_and_run_python")
    if not hasattr(args, 'content') or not args.content:
        raise ValueError("🚨 MISSING PARAMETER: content is required for write_file_and_run_python")
    if not args.content.strip():
        raise ValueError("🚨 EMPTY CONTENT: content parameter cannot be empty or whitespace-only")

    file_path = args.file_path
    content = args.content

    # Log pre-dispatch phase to retry ledger
    try:
        log_pre_dispatch("write_file_and_run_python", {"file_path": file_path, "content_length": len(content)})
    except Exception as e:
        # Log validation error if args processing fails
        log_pre_dispatch("write_file_and_run_python", {"raw_args": {"file_path": file_path, "content": content[:100]}}, str(e))
        raise

    # Log tool call start
    state.log_react_event("tool_call_start", {
        "tool_name": "write_file_and_run_python",
        "args": {"file_path": file_path, "content_length": len(content)}
    })

    try:
        # Step 1: Write the file
        full_path = state.workspace_dir / file_path

        # Ensure directory exists
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Step 2: Execute the script immediately
        # Create logs directory in parent workspace
        logs_dir = state.workspace_dir.parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique task ID for this execution
        task_id = f"{int(time.time())}-{file_path.replace('/', '_').replace('.py', '')}"
        stdout_log = (logs_dir / f"stdout-{task_id}.log").resolve()
        stderr_log = (logs_dir / f"stderr-{task_id}.log").resolve()

        # Change to workspace directory for execution
        original_cwd = os.getcwd()
        os.chdir(state.workspace_dir)

        try:
            # Use uv venv Python - no fallbacks
            if not os.environ.get('VIRTUAL_ENV'):
                raise RuntimeError("VIRTUAL_ENV not set - uv venv must be activated")

            venv_python = os.path.join(os.environ['VIRTUAL_ENV'], 'bin', 'python')
            if not os.path.exists(venv_python):
                # Windows path
                venv_python = os.path.join(os.environ['VIRTUAL_ENV'], 'Scripts', 'python.exe')

            if not os.path.exists(venv_python):
                raise RuntimeError(f"Python not found in venv: {os.environ['VIRTUAL_ENV']}")

            result = subprocess.run(
                [venv_python, file_path],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                env=os.environ.copy()  # Pass environment variables including VIRTUAL_ENV
            )

            # Write stdout and stderr to separate log files
            try:
                if result.stdout:
                    with open(stdout_log, 'w', encoding='utf-8') as f:
                        f.write(result.stdout)
                if result.stderr:
                    with open(stderr_log, 'w', encoding='utf-8') as f:
                        f.write(result.stderr)
            except IOError as log_error:
                # Log file creation failed, but continue execution
                print(f"Warning: Could not write log files: {log_error}")
                pass

            success = result.returncode == 0
            end_time = time.time()
            duration = end_time - start_time

            # Create tail excerpts for large outputs (last 500 chars)
            stdout_tail = None
            stderr_tail = None

            if result.stdout and len(result.stdout) > 500:
                stdout_tail = "..." + result.stdout[-500:]
            elif result.stdout:
                stdout_tail = result.stdout

            if result.stderr and len(result.stderr) > 500:
                stderr_tail = "..." + result.stderr[-500:]
            elif result.stderr:
                stderr_tail = result.stderr

            # Log detailed script execution results
            state.log_react_event("script_execution_detailed", {
                "file_path": file_path,
                "content_length": len(content),
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": success,
                "duration": end_time - start_time
            })

            # Log tool call completion
            state.log_react_event("tool_call_complete", {
                "tool_name": "write_file_and_run_python",
                "success": success,
                "file_written": file_path,
                "content_length": len(content),
                "exit_code": result.returncode,
                "task_id": task_id,
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration
            })

            if not success:
                error_msg = f"Script failed with exit code {result.returncode}"
                state.add_tool_timing("write_file_and_run_python", start_time, end_time, False, error_msg)
            else:
                state.add_tool_timing("write_file_and_run_python", start_time, end_time, True)

            # Log execution phase to retry ledger
            log_execution(
                tool_name="write_file_and_run_python",
                exit_code=result.returncode,
                duration_s=duration,
                stderr_path=str(stderr_log) if result.stderr else None,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail
            )

            # Log post-process phase to retry ledger - tool result successfully returned
            log_post_process("write_file_and_run_python")

            # Return structured result instead of text blob
            return WriteRunResult(
                file_path=file_path,
                exit_code=result.returncode,
                stdout_log=str(stdout_log),
                stderr_log=str(stderr_log) if result.stderr else None,
                duration_s=duration,
                success=success,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
                task_id=task_id,
                content_length=len(content)
            )

        finally:
            os.chdir(original_cwd)

    except subprocess.TimeoutExpired:
        end_time = time.time()
        duration = end_time - start_time
        error_msg = "Script execution timed out (5 minutes)"
        state.add_tool_timing("write_file_and_run_python", start_time, end_time, False, error_msg)

        return WriteRunResult(
            file_path=file_path,
            exit_code=-1,
            stdout_log="",
            stderr_log=None,
            duration_s=duration,
            success=False,
            stdout_tail=None,
            stderr_tail=error_msg,
            task_id="timeout",
            content_length=len(content)
        )

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        error_msg = f"Error in write_file_and_run_python: {str(e)}"
        state.add_tool_timing("write_file_and_run_python", start_time, end_time, False, error_msg)

        return WriteRunResult(
            file_path=file_path,
            exit_code=-1,
            stdout_log="",
            stderr_log=None,
            duration_s=duration,
            success=False,
            stdout_tail=None,
            stderr_tail=error_msg,
            task_id="error",
            content_length=len(content)
        )


def create_agent(model_name: str, template_content: str) -> Agent:
    """Create a PydanticAI agent with the specified model and tools."""

    # Determine model provider and create appropriate model
    if model_name.startswith("anthropic:"):
        model = AnthropicModel(model_name.replace("anthropic:", ""))
    elif model_name.startswith("openai:"):
        model = OpenAIModel(model_name.replace("openai:", ""))
    elif model_name.startswith("ollama:"):
        # Local Ollama model via OpenAI-compatible API
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.ollama import OllamaProvider

        ollama_model_name = model_name.replace("ollama:", "")
        model = OpenAIChatModel(
            model_name=ollama_model_name,
            provider=OllamaProvider(base_url='http://localhost:11434/v1')
        )
    else:
        # Default to Anthropic if no prefix
        model = AnthropicModel(model_name)

    # Create agent with tools and retry configuration
    agent = Agent(
        model=model,
        system_prompt=template_content,
        retries=5  # Increased from 3 to handle complex temporal queries
    )

    # Register tools with timing - ONLY fused tool for script execution
    agent.tool(write_file_and_run_python)  # ONLY tool for script execution
    agent.tool(read_file)
    agent.tool(list_files)
    # REMOVED: write_file and run_python to force use of fused tool

    return agent


def load_template(template_path: str, query: str, dataset_description: str, **kwargs) -> str:
    """Load and render the Jinja2 template."""
    template_file = Path(template_path)
    if not template_file.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    # Load detailed dataset analysis if available
    dataset_analysis_path = Path("docs/2025_09_22_17_31_preliminary_data_analysis.md")
    dataset_analysis = ""
    if dataset_analysis_path.exists():
        dataset_analysis = dataset_analysis_path.read_text(encoding='utf-8')

    template = Template(template_file.read_text(encoding='utf-8'))

    return template.render(
        query=query,
        dataset_description=dataset_description,
        dataset_analysis=dataset_analysis,
        **kwargs
    )


def load_rubric(analysis_type: str) -> Dict[str, Any]:
    """Load evaluation rubric for the specified analysis type."""
    rubric_path = Path(f"config/rubrics/{analysis_type}_analysis_rubric.yaml")
    if not rubric_path.exists():
        # Fallback to factual rubric if specific one doesn't exist
        rubric_path = Path("config/rubrics/factual_analysis_rubric.yaml")
        if not rubric_path.exists():
            # Return empty rubric if no rubric files exist
            return {}

    with open(rubric_path, 'r') as f:
        return yaml.safe_load(f)


def select_template_by_analysis_type(query_data: Dict[str, Any], default_template: str) -> str:
    """Select appropriate template based on analysis type."""
    analysis_type = query_data.get("analysis_type", "data_analysis")

    template_map = {
        "factual": "templates/factual_analysis_agent_prompt.txt",
        "pattern": "templates/pattern_analysis_agent_prompt.txt",
        "causal": "templates/causal_analysis_agent_prompt.txt"
    }

    selected_template = template_map.get(analysis_type, default_template)

    # Verify template exists, fallback to default if not
    if not Path(selected_template).exists():
        print(f"Template {selected_template} not found, using default: {default_template}")
        return default_template

    return selected_template


def setup_workspace(base_workspace: Optional[str], query_id: str) -> Path:
    """Set up workspace directory for the agent run."""
    if base_workspace:
        workspace_root = Path(base_workspace)
    else:
        workspace_root = Path(".runs")

    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_uuid = str(uuid.uuid4())[:8]
    workspace_dir = workspace_root / f"{timestamp}-{run_uuid}"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Create workspace subdirectory for analysis files
    (workspace_dir / "workspace").mkdir(exist_ok=True)

    # Create symlink to data directory if it exists
    data_dir = Path("data")
    if data_dir.exists():
        symlink_path = workspace_dir / "workspace" / "data"
        if not symlink_path.exists():
            symlink_path.symlink_to(data_dir.absolute())

    return workspace_dir


def main():
    parser = argparse.ArgumentParser(description="Run PydanticAI-based data analysis agent")
    parser.add_argument("--task", required=True, help="Query to analyze")
    parser.add_argument("--query-id", required=True, help="Query identifier")
    parser.add_argument("--template", default="templates/data_analysis_agent_prompt.txt",
                       help="Template file path")
    parser.add_argument("--model", default="anthropic:claude-sonnet-4-20250514",
                       help="Model to use")
    parser.add_argument("--max-tools", type=int, default=8,
                       help="Maximum tool calls (reduced for efficiency)")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Timeout per tool execution")
    parser.add_argument("--workspace", help="Base workspace directory")
    parser.add_argument("--config", default="config/config.yaml",
                       help="Configuration file")
    parser.add_argument("--console-updates", action="store_true",
                       help="Enable console update messages")

    args = parser.parse_args()

    # Load configuration
    config = {}
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) or {}

    # Setup workspace
    workspace_dir = setup_workspace(args.workspace, args.query_id)

    # Initialize timing
    total_start_time = time.time()

    # Determine analysis type and load rubric
    query_data = {"query": args.task}  # Basic query data structure
    analysis_type = query_data.get("analysis_type", "factual")  # Default to factual for most queries
    rubric = load_rubric(analysis_type)

    # Create agent state
    state = AgentState(
        workspace_dir=workspace_dir / "workspace",
        query=args.task,
        query_id=args.query_id,
        dataset_description=config.get("dataset", {}).get("description", "Natural gas pipeline transportation data"),
        analysis_type=analysis_type,
        rubric=rubric,
        console_updates_enabled=args.console_updates,
        total_start_time=total_start_time,
        react_log_path=workspace_dir / "workspace" / "react_log.jsonl"
    )

    # Determine template based on query analysis type if provided
    template_path = select_template_by_analysis_type({"analysis_type": analysis_type}, args.template)

    if state.console_updates_enabled:
        print(f"Starting analysis with template: {template_path}")
        print(f"Workspace: {workspace_dir}")

    try:
        # Load and render template
        template_content = load_template(
            template_path,
            args.task,
            state.dataset_description,
            analysis_type=analysis_type,
            rubric=rubric
        )

        # Create agent
        if state.console_updates_enabled:
            print(f"Creating agent with model: {args.model}")
        agent = create_agent(args.model, template_content)

        # Log task start
        state.log_react_event("task_start", {
            "query": args.task,
            "template": template_path,
            "model": args.model
        })

        # Run agent
        if state.console_updates_enabled:
            print("Running agent analysis...")

        # Log LLM input
        state.log_react_event("llm_input", {
            "system_prompt_length": len(template_content),
            "user_query": args.task
        })

        # Create custom LogEvent handler to capture all LLM interactions
        class ReActStepLogger:
            def __init__(self, state):
                self.state = state
                self.step_count = 0

            def log_complete_conversation_flow(self, messages):
                """Log the complete conversation history with detailed breakdown"""
                self.state.log_react_event("complete_conversation_flow", {
                    "total_messages": len(messages),
                    "conversation_breakdown": self._extract_message_breakdown(messages)
                })

                # Log each message with full details
                for i, message in enumerate(messages):
                    self._log_single_message(i, message)

            def _extract_message_breakdown(self, messages):
                """Extract detailed breakdown of all message parts"""
                breakdown = []
                for i, message in enumerate(messages):
                    msg_info = {
                        "message_index": i,
                        "message_type": type(message).__name__,
                        "content_preview": str(message)[:200] + "..." if len(str(message)) > 200 else str(message)
                    }

                    if hasattr(message, 'parts'):
                        msg_info["parts_count"] = len(message.parts)
                        msg_info["parts"] = []

                        for j, part in enumerate(message.parts):
                            part_info = {
                                "part_index": j,
                                "part_type": type(part).__name__,
                                "timestamp": getattr(part, 'timestamp', None)
                            }

                            # Extract content based on part type
                            if hasattr(part, 'content'):
                                content = part.content
                                part_info["content"] = content[:500] + "..." if len(content) > 500 else content
                            if hasattr(part, 'tool_name'):
                                part_info["tool_name"] = part.tool_name
                                part_info["tool_args"] = getattr(part, 'args', {})
                                part_info["tool_call_id"] = getattr(part, 'tool_call_id', None)

                            msg_info["parts"].append(part_info)

                    breakdown.append(msg_info)
                return breakdown

            def _log_single_message(self, index, message):
                """Log a single message with full details"""
                self.state.log_react_event("message_detail", {
                    "message_index": index,
                    "message_type": type(message).__name__,
                    "full_content": str(message),
                    "parts_available": hasattr(message, 'parts'),
                    "parts_count": len(message.parts) if hasattr(message, 'parts') else 0
                })

        # Use agent with comprehensive logging
        logger = ReActStepLogger(state)

        # Note: PydanticAI doesn't have direct hooks, so we'll wrap the run_sync
        # We'll capture this through tool execution logs and final result
        state.log_react_event("agent_execution_start", {
            "task": args.task,
            "timestamp": datetime.now().isoformat()
        })

        try:
            # Initialize retry ledger for this query
            retry_ledger = RetryLedger(args.query_id, workspace_dir)
            set_current_ledger(retry_ledger)

            # Enhanced debugging: Track agent execution flow
            execution_debug = {
                "agent_start_time": time.time(),
                "expected_steps": ["initial_request", "first_tool_call", "tool_result", "second_tool_call?", "final_response"],
                "actual_steps": []
            }

            state.log_react_event("debug_agent_start", execution_debug)
            print(f"🔍 DEBUG: Starting agent with task: {args.task[:100]}...")

            # COMPREHENSIVE PYDANTIC AI DEBUGGING - Hook into internal mechanisms
            from pydantic_ai import Agent
            import logging
            import sys

            # Create detailed PydanticAI logger
            pydantic_logger = logging.getLogger('pydantic_ai_debug')
            pydantic_logger.setLevel(logging.DEBUG)

            # Create handler to capture all PydanticAI internal activity
            log_handler = logging.StreamHandler(sys.stdout)
            log_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('🔍 PYDANTIC[%(levelname)s]: %(message)s')
            log_handler.setFormatter(formatter)
            pydantic_logger.addHandler(log_handler)

            # Hook into PydanticAI's logging
            for logger_name in ['pydantic_ai', 'pydantic_ai.agent', 'pydantic_ai.tools', 'pydantic_ai.models']:
                logger = logging.getLogger(logger_name)
                logger.setLevel(logging.DEBUG)
                logger.addHandler(log_handler)

            # Monkey patch PydanticAI's critical methods to add debugging + parameter fixing
            original_run_sync = agent.run_sync

            def debug_run_sync(task, **kwargs):
                print(f"🔍 PYDANTIC ENTRY: run_sync called with task: {task[:100]}...")
                try:
                    result = original_run_sync(task, **kwargs)
                    print(f"🔍 PYDANTIC SUCCESS: run_sync returned result type: {type(result)}")
                    print(f"🔍 PYDANTIC SUCCESS: result has {len(result.all_messages()) if hasattr(result, 'all_messages') else 'unknown'} messages")
                    return result
                except Exception as e:
                    # AGGRESSIVE FIX: Detect and handle missing content parameter
                    if ("ValidationError" in str(type(e).__name__) and
                        "content" in str(e) and
                        "Field required" in str(e) and
                        "write_file_and_run_python" in str(e)):

                        print(f"🚨 DETECTED MISSING CONTENT PARAMETER!")
                        print(f"🚨 PydanticAI generated incomplete tool call")
                        print(f"🔧 ATTEMPTING PARAMETER RECONSTRUCTION...")

                        # This is a fundamental PydanticAI issue that we can't easily fix
                        # without major framework changes. For now, log it clearly.
                        print(f"❌ CANNOT AUTO-FIX: This requires PydanticAI framework repair")

                    print(f"🔍 PYDANTIC FAILURE: run_sync raised {type(e).__name__}: {str(e)}")
                    print(f"🔍 PYDANTIC FAILURE: Exception args: {e.args}")
                    if hasattr(e, '__traceback__'):
                        import traceback
                        print(f"🔍 PYDANTIC FAILURE: Traceback:\n{traceback.format_exc()}")
                    raise

            agent.run_sync = debug_run_sync

            # DEEP HOOK: Intercept tool execution at the lowest level
            # Find and hook PydanticAI's tool calling mechanism
            try:
                # Hook into the agent's tool registry if available
                if hasattr(agent, '_tools'):
                    print(f"🔍 PYDANTIC TOOLS: Found {len(agent._tools)} registered tools")
                    for tool_name, tool_func in agent._tools.items():
                        print(f"🔍 PYDANTIC TOOLS: - {tool_name}")

                        # Wrap each tool function with debugging
                        def create_tool_wrapper(orig_func, name):
                            def wrapped_tool(*args, **kwargs):
                                print(f"🔍 TOOL ENTRY: {name} called with args={len(args)}, kwargs={list(kwargs.keys())}")
                                try:
                                    result = orig_func(*args, **kwargs)
                                    print(f"🔍 TOOL SUCCESS: {name} returned {type(result)}: {str(result)[:200]}...")
                                    return result
                                except Exception as e:
                                    print(f"🔍 TOOL FAILURE: {name} raised {type(e).__name__}: {str(e)}")
                                    raise
                            return wrapped_tool

                        agent._tools[tool_name] = create_tool_wrapper(tool_func, tool_name)

                # Also try to hook the model interface
                if hasattr(agent, '_model'):
                    print(f"🔍 PYDANTIC MODEL: Using model type {type(agent._model)}")

                    # Hook model completion if possible
                    if hasattr(agent._model, 'request'):
                        original_request = agent._model.request

                        def debug_model_request(*args, **kwargs):
                            print(f"🔍 MODEL REQUEST: Called with {len(args)} args, {len(kwargs)} kwargs")
                            try:
                                result = original_request(*args, **kwargs)
                                print(f"🔍 MODEL SUCCESS: Request returned {type(result)}")
                                return result
                            except Exception as e:
                                print(f"🔍 MODEL FAILURE: Request raised {type(e).__name__}: {str(e)}")
                                raise

                        agent._model.request = debug_model_request

            except Exception as debug_error:
                print(f"🔍 DEBUG SETUP ERROR: {debug_error}")

            # Execute agent and capture result with enhanced monitoring
            start_execution = time.time()
            print(f"🔍 STARTING PYDANTIC EXECUTION...")
            result = agent.run_sync(args.task, deps=state)
            end_execution = time.time()
            print(f"🔍 PYDANTIC EXECUTION COMPLETED SUCCESSFULLY")

            execution_debug["agent_end_time"] = end_execution
            execution_debug["total_execution_time"] = end_execution - start_execution
            execution_debug["agent_completed"] = True

            print(f"🔍 DEBUG: Agent completed in {end_execution - start_execution:.2f}s")
            state.log_react_event("debug_agent_complete", execution_debug)

            # Log ALL message history with detailed breakdown
            all_messages = result.all_messages()
            logger = ReActStepLogger(state)
            logger.log_complete_conversation_flow(all_messages)

            # ENHANCED DEBUG: Analyze conversation flow for tool call patterns
            tool_call_analysis = {
                "total_messages": len(all_messages),
                "tool_calls_detected": 0,
                "tool_results_detected": 0,
                "final_response_detected": False,
                "message_breakdown": []
            }

            for i, message in enumerate(all_messages):
                msg_analysis = {
                    "index": i,
                    "type": type(message).__name__,
                    "content_preview": str(message)[:200] + "..." if len(str(message)) > 200 else str(message)
                }

                # Check for tool call patterns
                msg_str = str(message).lower()
                if "write_file_and_run_python" in msg_str:
                    if "tool_call" in msg_str or "function_call" in msg_str:
                        tool_call_analysis["tool_calls_detected"] += 1
                        msg_analysis["is_tool_call"] = True
                    elif "result" in msg_str or "output" in msg_str:
                        tool_call_analysis["tool_results_detected"] += 1
                        msg_analysis["is_tool_result"] = True

                # Check for final response patterns
                if "response.json" in msg_str or "confidence" in msg_str:
                    tool_call_analysis["final_response_detected"] = True
                    msg_analysis["is_final_response"] = True

                tool_call_analysis["message_breakdown"].append(msg_analysis)

            print(f"🔍 TOOL CALL ANALYSIS:")
            print(f"   Total messages: {tool_call_analysis['total_messages']}")
            print(f"   Tool calls detected: {tool_call_analysis['tool_calls_detected']}")
            print(f"   Tool results detected: {tool_call_analysis['tool_results_detected']}")
            print(f"   Final response detected: {tool_call_analysis['final_response_detected']}")

            state.log_react_event("debug_tool_call_analysis", tool_call_analysis)

            # Calculate token usage and costs from conversation messages
            total_input_tokens = 0
            total_output_tokens = 0
            model_name = args.model

            for i, message in enumerate(all_messages):
                # First, try to get usage information directly from PydanticAI
                if hasattr(message, 'usage') and message.usage:
                    usage = message.usage
                    if hasattr(usage, 'input_tokens'):
                        total_input_tokens += getattr(usage, 'input_tokens', 0)
                    if hasattr(usage, 'output_tokens'):
                        total_output_tokens += getattr(usage, 'output_tokens', 0)
                    continue

                # Fallback to manual token counting for message parts
                if hasattr(message, 'parts'):
                    # ModelRequest or ModelResponse with parts
                    for part in message.parts:
                        part_text = ""
                        if hasattr(part, 'content'):
                            part_text = str(part.content)
                        else:
                            part_text = str(part)

                        if part_text.strip() and len(part_text) < 50000:  # Reasonable text length
                            tokens = count_tokens(part_text, model_name)

                            # Classify based on message type and part type
                            if 'ModelRequest' in str(type(message)):
                                total_input_tokens += tokens
                            elif 'ModelResponse' in str(type(message)):
                                total_output_tokens += tokens
                            else:
                                total_input_tokens += tokens  # Default to input

            # Calculate cost based on token usage
            cost_info = calculate_cost(total_input_tokens, total_output_tokens, model_name)

            # Store cost information in state for later retrieval
            state.cost_info = {
                'model': model_name,
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens,
                'total_tokens': total_input_tokens + total_output_tokens,
                'total_cost': cost_info['total_cost'],
                'cost_breakdown': cost_info
            }

            # Log final execution summary
            state.log_react_event("agent_execution_complete", {
                "final_result": str(result),
                "total_messages_exchanged": len(all_messages),
                "success": True,
                "timestamp": datetime.now().isoformat()
            })

            # Save complete conversation log to separate file
            conversation_log = {
                "query_id": args.query_id,
                "query": args.task,
                "total_messages": len(all_messages),
                "complete_conversation": [str(msg) for msg in all_messages],
                "detailed_breakdown": logger._extract_message_breakdown(all_messages)
            }

            conversation_file = workspace_dir / "complete_conversation_log.json"
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_log, f, indent=2, default=str)

        except Exception as e:
            # COMPREHENSIVE FAILURE ANALYSIS - Capture every detail
            print(f"🔍 EXCEPTION CAUGHT: {type(e).__name__}: {str(e)}")

            # Check if this is the specific validation error we're trying to fix
            if ("ValidationError" in str(type(e).__name__) and
                "content" in str(e) and
                "Field required" in str(e)):
                print(f"🚨 PARAMETER VALIDATION ERROR DETECTED!")
                print(f"🚨 Agent failed to provide required 'content' parameter")
                print(f"🚨 This is NOT a retry issue - it's incomplete tool arguments")

            # Enhanced error logging to capture exact failure details
            error_details = {
                "error": str(e),
                "error_type": type(e).__name__,
                "success": False,
                "timestamp": datetime.now().isoformat(),
                "traceback": str(e.__traceback__) if hasattr(e, '__traceback__') else None
            }

            # CRITICAL: Check if this is the success-interpreted-as-failure case
            retry_ledger = get_current_ledger()
            if retry_ledger:
                retries_burned = sum(a.retry_delta for a in retry_ledger.attempts)
                print(f"🔍 RETRY ANALYSIS: {retries_burned} retries actually burned")

                # If no retries burned but still failing, this is the bug
                if retries_burned == 0 and "exceeded max retries" in str(e):
                    print(f"🚨 SUCCESS-AS-FAILURE BUG DETECTED!")
                    print(f"🚨 Tool executions successful but PydanticAI reports retry failure")

                    # Capture detailed state at failure
                    error_details["bug_detected"] = "success_interpreted_as_failure"
                    error_details["actual_retries_burned"] = retries_burned
                    error_details["tool_executions"] = [
                        {
                            "attempt": a.attempt_number,
                            "phase": a.phase.value,
                            "exit_code": a.exit_code,
                            "retry_delta": a.retry_delta,
                            "tool_name": a.tool_name
                        } for a in retry_ledger.attempts
                    ]

            # Try to extract more information from PydanticAI's internal state
            try:
                if hasattr(e, '__context__'):
                    print(f"🔍 EXCEPTION CONTEXT: {e.__context__}")
                if hasattr(e, '__cause__'):
                    print(f"🔍 EXCEPTION CAUSE: {e.__cause__}")
                if hasattr(e, '__traceback__'):
                    import traceback
                    full_traceback = traceback.format_exc()
                    print(f"🔍 FULL TRACEBACK:\n{full_traceback}")
                    error_details["full_traceback"] = full_traceback

                # Try to inspect the agent's state at failure
                if 'agent' in locals():
                    agent_state = {}
                    for attr in ['_model', '_tools', '_system_prompt']:
                        if hasattr(agent, attr):
                            agent_state[attr] = str(getattr(agent, attr))[:200]
                    error_details["agent_state_at_failure"] = agent_state

            except Exception as debug_error:
                print(f"🔍 DEBUG ERROR: {debug_error}")
                error_details["debug_error"] = str(debug_error)

            # Add PydanticAI specific error details if available
            if hasattr(e, 'args') and e.args:
                error_details["error_args"] = list(e.args)

            # Enhanced debugging for retry errors
            if "exceeded max retries" in str(e):
                error_details["failure_type"] = "tool_retry_limit_exceeded"

                # Get current tool execution count from state
                tool_execution_count = len([t for t in state.tool_timings if t.tool_name == "write_file_and_run_python"])

                error_details["tool_execution_count"] = tool_execution_count
                error_details["tool_timings"] = [
                    {
                        "tool": t.tool_name,
                        "success": t.success,
                        "duration": t.duration
                    } for t in state.tool_timings
                ]

                # Check what files were created
                workspace_files = []
                try:
                    for file_path in state.workspace_dir.rglob("*"):
                        if file_path.is_file():
                            workspace_files.append(str(file_path.relative_to(state.workspace_dir)))
                except Exception:
                    pass

                error_details["workspace_files"] = workspace_files

                print(f"🔍 RETRY ERROR DEBUG:")
                print(f"   Tool executions: {tool_execution_count}")
                print(f"   Last tool success: {state.tool_timings[-1].success if state.tool_timings else 'None'}")
                print(f"   Workspace files: {len(workspace_files)}")
                print(f"   Error details: {str(e)}")

                # Print retry summary from ledger if this was a retry-related failure
                print_retry_summary(str(e))

            state.log_react_event("agent_execution_complete", error_details)

            # Also write detailed error to a separate file for investigation
            error_log_file = state.workspace_dir / "detailed_error_log.json"
            try:
                with open(error_log_file, 'w', encoding='utf-8') as f:
                    json.dump(error_details, f, indent=2, default=str)
            except Exception:
                pass  # Don't fail on error log writing

            raise

        # Agent MUST create response.json - no fallbacks
        agent_response_file = state.workspace_dir / "response.json"
        response_file = state.workspace_dir / "response.json"

        if not agent_response_file.exists():
            raise RuntimeError(f"Agent failed to create required response.json file at {agent_response_file}")

        try:
            with open(agent_response_file, 'r', encoding='utf-8') as f:
                agent_response = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise RuntimeError(f"Agent created invalid response.json file: {e}")

        # Validate required fields
        required_fields = ["analysis_type", "answer", "methodology_explanation", "evidence_linkage", "limitations_uncertainties", "confidence"]
        missing_fields = [field for field in required_fields if field not in agent_response]
        if missing_fields:
            raise RuntimeError(f"Agent response.json missing required fields: {missing_fields}")

        # Enhance with metadata and copy to final location
        agent_response.update({
            "query_id": args.query_id,
            "query": args.task,
            "plain_text_response": str(result),
            "timestamp": datetime.now().isoformat(),
            "model_used": args.model,
            "template_used": template_path
        })

        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump(agent_response, f, indent=2, default=str)

        print(f"✅ Agent successfully created comprehensive response.json with {len(agent_response)} fields")

        # Get timing summary
        timing_summary = state.get_timing_summary()

        # Save timing information
        timing_file = workspace_dir / "timing_breakdown.json"
        with open(timing_file, 'w') as f:
            json.dump(timing_summary, f, indent=2, default=str)

        # Save metadata
        metadata = {
            "query_id": args.query_id,
            "query": args.task,
            "template_used": template_path,
            "model": args.model,
            "timestamp": datetime.now().isoformat(),
            "workspace_dir": str(workspace_dir),
            "success": True,
            "timing_summary": timing_summary
        }

        # Add cost information if available
        if hasattr(state, 'cost_info') and state.cost_info:
            metadata["cost_info"] = state.cost_info
        metadata_file = workspace_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Print results
        print(f"\n{'='*60}")
        print("=== FINAL ANSWER ===")
        print(f"{'='*60}")
        print(str(result))

        print(f"\n{'='*60}")
        print("=== EXECUTION SUMMARY ===")
        print(f"{'='*60}")
        print(f"Run directory: {workspace_dir}")
        print(f"Total execution time: {timing_summary['total_elapsed_time']:.2f}s")
        print(f"Tool execution time: {timing_summary['total_tool_time']:.2f}s")
        print(f"Thinking time: {timing_summary['thinking_time']:.2f}s")
        print(f"Tool efficiency: {timing_summary['tool_efficiency']:.1%}")

        print(f"\n📊 TOOL BREAKDOWN:")
        for tool_name, tool_stats in timing_summary['tool_breakdown'].items():
            print(f"  {tool_name}: {tool_stats['count']} calls, "
                  f"{tool_stats['total_duration']:.2f}s total, "
                  f"{tool_stats['successful_calls']} success, "
                  f"{tool_stats['failed_calls']} failed")

        # Print cost information if available
        if hasattr(state, 'cost_info') and state.cost_info:
            cost_info = state.cost_info
            print(f"\n💰 COST BREAKDOWN:")
            print(f"  Model: {cost_info.get('model', 'unknown')}")
            print(f"  Input tokens: {cost_info['input_tokens']:,}")
            print(f"  Output tokens: {cost_info['output_tokens']:,}")
            print(f"  Total tokens: {cost_info['total_tokens']:,}")
            print(f"  Total cost: ${cost_info['total_cost']:.4f}")

        return 0

    except Exception as e:
        print(f"\n❌ Agent execution failed: {str(e)}")
        print(f"   Error type: {type(e).__name__}")

        # Enhanced error details for debugging
        if "exceeded max retries" in str(e):
            print(f"   🔍 RETRY LIMIT ERROR: This indicates a tool execution failed and hit PydanticAI's retry limit")
            print(f"   📊 Tool executions so far: {len(state.tool_timings)}")
            print(f"   🛠️  Last successful tool: {state.tool_timings[-1].tool_name if state.tool_timings else 'None'}")

        # Still save partial timing information
        timing_summary = state.get_timing_summary()
        timing_file = workspace_dir / "timing_breakdown.json"
        with open(timing_file, 'w') as f:
            json.dump(timing_summary, f, indent=2, default=str)

        # Save comprehensive error details for debugging
        error_debug = {
            "error_message": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat(),
            "tool_execution_count": len(state.tool_timings),
            "tool_timings": [{"tool": t.tool_name, "success": t.success, "duration": t.duration} for t in state.tool_timings],
            "workspace_files": []
        }

        # List files created so far
        try:
            for file_path in workspace_dir.rglob("*"):
                if file_path.is_file():
                    error_debug["workspace_files"].append(str(file_path.relative_to(workspace_dir)))
        except Exception:
            pass

        error_debug_file = workspace_dir / "execution_failure_debug.json"
        try:
            with open(error_debug_file, 'w') as f:
                json.dump(error_debug, f, indent=2, default=str)
        except Exception:
            pass

        return 1


if __name__ == "__main__":
    exit(main())
