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

import yaml
from dotenv import load_dotenv
from jinja2 import Template
from pydantic import BaseModel
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


def write_file_and_run_python(ctx: RunContext[AgentState], file_path: str, content: str) -> str:
    """Write a Python script file and immediately execute it. This fused operation reduces tool calls and avoids rewrites."""
    state = ctx.deps
    start_time = time.time()

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
            result = subprocess.run(
                ["python", file_path],
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

            success = result.returncode == 0
            end_time = time.time()

            # Create comprehensive output message
            output = f"📝 File written: {file_path} ({len(content)} characters)\n"
            output += f"🚀 Script executed: Exit code {result.returncode}\n"
            output += f"⏱️  Execution time: {end_time - start_time:.2f}s\n"

            if result.stdout:
                output += f"\n📤 STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"\n🚨 STDERR:\n{result.stderr}\n"

            if success:
                output += "✅ Script completed successfully"
            else:
                output += f"❌ Script failed with exit code {result.returncode}"

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
                "duration": end_time - start_time
            })

            if not success:
                error_msg = f"Script failed with exit code {result.returncode}"
                state.add_tool_timing("write_file_and_run_python", start_time, end_time, False, error_msg)
            else:
                state.add_tool_timing("write_file_and_run_python", start_time, end_time, True)

            return output

        finally:
            os.chdir(original_cwd)

    except subprocess.TimeoutExpired:
        end_time = time.time()
        error_msg = "Script execution timed out (5 minutes)"
        state.add_tool_timing("write_file_and_run_python", start_time, end_time, False, error_msg)
        return f"📝 File written: {file_path}\n❌ {error_msg}"

    except Exception as e:
        end_time = time.time()
        error_msg = f"Error in write_file_and_run_python: {str(e)}"
        state.add_tool_timing("write_file_and_run_python", start_time, end_time, False, error_msg)
        return f"❌ {error_msg}"


def create_agent(model_name: str, template_content: str) -> Agent:
    """Create a PydanticAI agent with the specified model and tools."""

    # Determine model provider and create appropriate model
    if model_name.startswith("anthropic:"):
        model = AnthropicModel(model_name.replace("anthropic:", ""))
    elif model_name.startswith("openai:"):
        model = OpenAIModel(model_name.replace("openai:", ""))
    else:
        # Default to Anthropic if no prefix
        model = AnthropicModel(model_name)

    # Create agent with tools
    agent = Agent(
        model=model,
        system_prompt=template_content
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
            # Execute agent and capture result
            result = agent.run_sync(args.task, deps=state)

            # Log ALL message history with detailed breakdown
            all_messages = result.all_messages()
            logger = ReActStepLogger(state)
            logger.log_complete_conversation_flow(all_messages)

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
            state.log_react_event("agent_execution_complete", {
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            })
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

        return 0

    except Exception as e:
        print(f"\n❌ Agent execution failed: {str(e)}")

        # Still save partial timing information
        timing_summary = state.get_timing_summary()
        timing_file = workspace_dir / "timing_breakdown.json"
        with open(timing_file, 'w') as f:
            json.dump(timing_summary, f, indent=2, default=str)

        return 1


if __name__ == "__main__":
    exit(main())
