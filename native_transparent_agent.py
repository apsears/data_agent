#!/usr/bin/env python3
"""
Native Transparent Agent Framework - Using Anthropic's built-in tool calling API

This version leverages the native Anthropic tool calling functionality instead of manual XML parsing.
Much cleaner, more reliable, and follows best practices.
"""

import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

import anthropic
import tiktoken
import litellm
from tools.utils.retry_ledger import RetryLedger, set_current_ledger, log_execution, log_pre_dispatch, log_post_process

# Configure litellm to drop unsupported parameters for GPT-5 models
litellm.drop_params = True


def calculate_token_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for model usage based on token counts and pricing data."""
    # Determine which pricing file to use based on model name
    if model_name.startswith("openai:") or model_name.startswith("gpt-"):
        pricing_file = Path("config/openai_pricing.tsv")
    else:
        pricing_file = Path("config/anthropic_pricing.tsv")

    import pandas as pd
    df = pd.read_csv(pricing_file, sep='\t')

    model_key = model_name.replace("anthropic:", "").replace("openai:", "")
    model_row = df[df['Model'] == model_key]

    if model_row.empty:
        raise ValueError(f"Pricing data not found for model: {model_name}")

    # Get input and output rates, handling both string ($0.25) and numeric (3.00) formats
    input_value = model_row.iloc[0]['Input']
    output_value = model_row.iloc[0]['Output']

    # Convert to float, removing $ if present
    if isinstance(input_value, str) and input_value.startswith('$'):
        input_rate = float(input_value.replace('$', ''))
    else:
        input_rate = float(input_value)

    if isinstance(output_value, str) and output_value.startswith('$'):
        output_rate = float(output_value.replace('$', ''))
    else:
        output_rate = float(output_value)

    return (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate


class ErrorCategory(Enum):
    """Precise error categorization for surgical debugging"""
    MODEL_REFUSAL = "model_refusal"
    PARSING_ERROR = "parsing_error"
    VALIDATION_ERROR = "validation_error"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT_ERROR = "timeout_error"
    SYSTEM_ERROR = "system_error"
    SCHEMA_ERROR = "schema_error"
    NETWORK_ERROR = "network_error"


@dataclass
class ToolResult:
    """Structured tool execution result"""
    success: bool
    content: str
    exit_code: Optional[int] = None
    duration: Optional[float] = None
    error_category: Optional[ErrorCategory] = None
    error_details: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_cost: Optional[float] = None

    @classmethod
    def success_result(cls, content: str, duration: float = None, input_tokens: int = None, output_tokens: int = None, total_cost: float = None) -> "ToolResult":
        return cls(success=True, content=content, duration=duration, input_tokens=input_tokens, output_tokens=output_tokens, total_cost=total_cost)

    @classmethod
    def error_result(cls, content: str, category: ErrorCategory, details: str = None, duration: float = None, input_tokens: int = None, output_tokens: int = None, total_cost: float = None) -> "ToolResult":
        return cls(success=False, content=content, error_category=category, error_details=details, duration=duration, input_tokens=input_tokens, output_tokens=output_tokens, total_cost=total_cost)


@dataclass
class AgentContext:
    """Context for agent execution"""
    workspace_dir: Path
    query: str
    query_id: str
    dataset_description: str
    analysis_type: str
    rubric: Dict[str, Any]
    config: Dict[str, Any]
    console_updates_enabled: bool = True
    react_log_path: Optional[Path] = None
    tool_timings: List[Dict[str, Any]] = None
    total_start_time: float = None

    def __post_init__(self):
        if self.tool_timings is None:
            self.tool_timings = []
        if self.total_start_time is None:
            self.total_start_time = time.time()

    def log_react_event(self, event_type: str, data: Dict[str, Any]):
        """Log a ReAct event to the react_log.jsonl file."""
        if self.react_log_path:
            try:
                self.react_log_path.parent.mkdir(parents=True, exist_ok=True)
                event = {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": event_type,
                    "query_id": self.query_id,
                    **data
                }

                # Display console message if present and console updates enabled
                if "console_message" in data and self.console_updates_enabled:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] {data['console_message']}")

                with open(self.react_log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(event, default=str) + '\n')
            except Exception as e:
                print(f"Warning: Could not write to react log: {e}")

    def console_update(self, message: str) -> str:
        """Print console update message if enabled."""
        if self.console_updates_enabled:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
        return message


class NativeToolExecutor:
    """Tool execution using native capabilities with full instrumentation"""

    def __init__(self, context: AgentContext):
        self.context = context

    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> ToolResult:
        """Execute tool with full visibility and instrumentation"""
        start_time = time.time()

        # Phase 1: Pre-dispatch validation
        validation_result = self._validate_tool_call(tool_name, tool_input)
        if not validation_result.success:
            log_pre_dispatch(tool_name, tool_input, validation_result.error_details)
            return validation_result

        log_pre_dispatch(tool_name, tool_input)

        # Phase 2: Execution
        execution_result = self._execute_tool_directly(tool_name, tool_input)

        # Log execution phase
        log_execution(
            tool_name=tool_name,
            exit_code=execution_result.exit_code or (0 if execution_result.success else 1),
            duration_s=execution_result.duration or (time.time() - start_time),
            stderr_path=None,
            stdout_tail=execution_result.content[-500:] if len(execution_result.content) > 500 else execution_result.content,
            stderr_tail=execution_result.error_details[-500:] if execution_result.error_details and len(execution_result.error_details) > 500 else execution_result.error_details
        )

        # Phase 3: Post-process
        log_post_process(tool_name)

        return execution_result

    def _validate_tool_call(self, tool_name: str, tool_input: Dict[str, Any]) -> ToolResult:
        """Validate tool call arguments"""
        if tool_name == "write_file_and_run_python":
            if 'file_path' not in tool_input or 'content' not in tool_input:
                missing_params = []
                if 'file_path' not in tool_input:
                    missing_params.append("file_path")
                if 'content' not in tool_input:
                    missing_params.append("content")

                detailed_error = f"""
🚨 TOOL CALL ERROR: write_file_and_run_python missing required parameter(s): {', '.join(missing_params)}

❌ What you provided: {list(tool_input.keys())}
✅ What is required: ["file_path", "content"]

🔧 CORRECT USAGE EXAMPLE:
{{
  "file_path": "002_final_analysis.py",
  "content": "#!/usr/bin/env python3\\nimport pandas as pd\\n# Your complete Python code here"
}}

⚠️  You MUST provide BOTH parameters:
- file_path: The filename to create
- content: The complete Python code as a string

Please retry with BOTH parameters included."""

                return ToolResult.error_result(
                    detailed_error,
                    ErrorCategory.VALIDATION_ERROR,
                    f"write_file_and_run_python missing required parameters: {', '.join(missing_params)}"
                )
        else:
            return ToolResult.error_result(
                f"Unknown tool: {tool_name}",
                ErrorCategory.VALIDATION_ERROR,
                f"Tool '{tool_name}' is not supported"
            )

        return ToolResult.success_result("Validation passed")

    def _get_versioned_filename(self, requested_path: str) -> str:
        """Generate a versioned filename to prevent overwrites.

        Args:
            requested_path: The filename the agent requested (e.g., "001_scout_analysis.py")

        Returns:
            Versioned filename (e.g., "001_scout_analysis_v001.py")
        """
        path_obj = Path(requested_path)
        base_name = path_obj.stem  # filename without extension
        extension = path_obj.suffix  # .py
        parent_dir = path_obj.parent

        # Check if file already exists, and if so, increment version
        version = 1
        while True:
            versioned_name = f"{base_name}_v{version:03d}{extension}"
            versioned_path = parent_dir / versioned_name
            full_path = self.context.workspace_dir / versioned_path

            if not full_path.exists():
                return str(versioned_path)
            version += 1

    def _execute_tool_directly(self, tool_name: str, tool_input: Dict[str, Any]) -> ToolResult:
        """Direct execution without framework abstraction"""
        if tool_name == "write_file_and_run_python":
            return self._execute_python_tool(tool_input)
        else:
            return ToolResult.error_result(
                f"Tool execution not implemented: {tool_name}",
                ErrorCategory.SYSTEM_ERROR
            )

    def _execute_python_tool(self, tool_input: dict) -> ToolResult:
        """Direct Python execution with full visibility and automatic versioning"""
        start_time = time.time()
        requested_file_path = tool_input['file_path']
        content = tool_input['content']

        try:
            # Step 1: Generate versioned filename to prevent overwrites
            actual_file_path = self._get_versioned_filename(requested_file_path)
            full_path = self.context.workspace_dir / actual_file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Log the actual filename used for audit trail
            print(f"📝 Created file: {actual_file_path} (requested: {requested_file_path})")

            # Step 2: Execute the script using the actual filename
            logs_dir = self.context.workspace_dir.parent / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            task_id = f"{int(time.time())}-{actual_file_path.replace('/', '_').replace('.py', '')}"
            stdout_log = (logs_dir / f"stdout-{task_id}.log").resolve()
            stderr_log = (logs_dir / f"stderr-{task_id}.log").resolve()

            # Use uv venv Python
            if not os.environ.get('VIRTUAL_ENV'):
                return ToolResult.error_result(
                    "VIRTUAL_ENV not set - uv venv must be activated",
                    ErrorCategory.SYSTEM_ERROR,
                    "Virtual environment not found"
                )

            venv_python = os.path.join(os.environ['VIRTUAL_ENV'], 'bin', 'python')
            if not os.path.exists(venv_python):
                # Windows path
                venv_python = os.path.join(os.environ['VIRTUAL_ENV'], 'Scripts', 'python.exe')

            if not os.path.exists(venv_python):
                return ToolResult.error_result(
                    f"Python not found in venv: {os.environ['VIRTUAL_ENV']}",
                    ErrorCategory.SYSTEM_ERROR,
                    "Python executable not found in virtual environment"
                )

            # Change to workspace directory for execution
            original_cwd = os.getcwd()
            os.chdir(self.context.workspace_dir)

            try:
                result = subprocess.run(
                    [venv_python, actual_file_path],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                    env=os.environ.copy()
                )

                # Write stdout and stderr to separate log files
                if result.stdout:
                    with open(stdout_log, 'w', encoding='utf-8') as f:
                        f.write(result.stdout)
                if result.stderr:
                    with open(stderr_log, 'w', encoding='utf-8') as f:
                        f.write(result.stderr)

                duration = time.time() - start_time
                success = result.returncode == 0

                # Create comprehensive output message with actual filename
                output = f"Successfully executed {actual_file_path} (originally requested: {requested_file_path}, exit code: {result.returncode}, duration: {duration:.1f}s)\n"
                output += f"📝 ACTUAL FILE CREATED: {actual_file_path}\n"
                output += f"🔄 REQUESTED FILE: {requested_file_path}\n\n"

                if result.stdout:
                    output += f"STDOUT:\n{result.stdout}\n"
                if result.stderr:
                    output += f"STDERR:\n{result.stderr}\n"

                if success:
                    return ToolResult.success_result(output, duration)
                else:
                    return ToolResult.error_result(
                        output,
                        ErrorCategory.EXECUTION_ERROR,
                        f"Script failed with exit code {result.returncode}",
                        duration
                    )

            finally:
                os.chdir(original_cwd)

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return ToolResult.error_result(
                "Script execution timed out (5 minutes)",
                ErrorCategory.TIMEOUT_ERROR,
                "Script execution exceeded 300 second timeout",
                duration
            )
        except Exception as e:
            duration = time.time() - start_time
            return ToolResult.error_result(
                f"Error in write_file_and_run_python: {str(e)}",
                ErrorCategory.SYSTEM_ERROR,
                str(e),
                duration
            )

    def _execute_read_file(self, tool_input: dict) -> ToolResult:
        """Read file contents"""
        start_time = time.time()
        file_path = tool_input['file_path']

        try:
            full_path = self.context.workspace_dir / file_path
            if not full_path.exists():
                return ToolResult.error_result(
                    f"File not found: {file_path}",
                    ErrorCategory.VALIDATION_ERROR,
                    f"File does not exist: {full_path}"
                )

            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            duration = time.time() - start_time
            return ToolResult.success_result(
                f"Content of {file_path}:\n{content}",
                duration
            )

        except Exception as e:
            duration = time.time() - start_time
            return ToolResult.error_result(
                f"Error reading file {file_path}: {str(e)}",
                ErrorCategory.SYSTEM_ERROR,
                str(e),
                duration
            )



class NativeReActExecutor:
    """Native ReAct loop using Anthropic's built-in tool calling"""

    def __init__(self, context: AgentContext, model_name: str, analyst_max_tokens: int = 16000):
        self.context = context
        self.model_name = model_name
        self.analyst_max_tokens = analyst_max_tokens
        self.client = anthropic.Anthropic()
        self.tool_executor = NativeToolExecutor(context)
        self.conversation_history = []

        # Define tools using native Anthropic schema
        self.tools = [
            {
                "name": "write_file_and_run_python",
                "description": "Write a Python script to a file and execute it in the workspace. CRITICAL: Both file_path AND content parameters are MANDATORY - you must provide the complete Python code in the content parameter.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the Python file to create (e.g., 'analysis.py') - REQUIRED"
                        },
                        "content": {
                            "type": "string",
                            "description": "Complete Python code content to write and execute - REQUIRED. Never omit this parameter. Must contain the full script code."
                        }
                    },
                    "required": ["file_path", "content"]
                }
            }
        ]

    def execute_react_cycle(self, system_prompt: str, user_query: str, max_iterations: int = 10) -> str:
        """Execute complete ReAct cycle using native tool calling"""

        self.conversation_history = [{"role": "user", "content": user_query}]

        self.context.log_react_event("react_cycle_start", {
            "max_iterations": max_iterations,
            "system_prompt_length": len(system_prompt),
            "user_query": user_query,
            "console_message": "🚀 Starting analysis cycle"
        })

        for iteration in range(max_iterations):
            self.context.log_react_event("react_iteration_start", {
                "iteration": iteration,
                "conversation_length": len(self.conversation_history)
            })

            try:
                # Console logging before LLM call
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                react_stage = f"ReAct Iteration {iteration + 1}/{max_iterations} - Analyst Reasoning"
                if self.context.console_updates_enabled:
                    print(f"[{timestamp}] 🧠 LLM CALL: {react_stage} | Model: {self.model_name}")

                # Make API call using unified LLM interface
                response = unified_llm_completion(
                    model=self.model_name,
                    messages=self.conversation_history,
                    system=system_prompt,
                    tools=self.tools,
                    max_tokens=self.analyst_max_tokens,
                    temperature=0.0
                )

                assistant_content = ""
                tool_results = []

                # Process response content
                for content_block in response["content"]:
                    if content_block["type"] == "text":
                        assistant_content += content_block["text"]
                    elif content_block["type"] == "tool_use":
                        # Execute tool using normalized structure
                        tool_name = content_block["name"]
                        tool_input = content_block["input"]
                        tool_use_id = content_block["id"]

                        self.context.log_react_event("tool_execution_start", {
                            "iteration": iteration,
                            "tool_name": tool_name,
                            "tool_input": tool_input,
                            "tool_use_id": tool_use_id
                        })

                        # Execute the tool
                        tool_result = self.tool_executor.execute_tool(tool_name, tool_input)

                        self.context.log_react_event("tool_execution_complete", {
                            "iteration": iteration,
                            "tool_name": tool_name,
                            "success": tool_result.success,
                            "result_length": len(tool_result.content),
                            "duration": tool_result.duration
                        })

                        # Add tool result to conversation
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": tool_result.content,
                            "is_error": not tool_result.success
                        })

                # Add assistant message to conversation
                assistant_message = {"role": "assistant", "content": []}
                if assistant_content:
                    assistant_message["content"].append({"type": "text", "text": assistant_content})

                # Add tool uses to assistant message
                if "tool_calls" in response and response["tool_calls"]:
                    for tool_call in response["tool_calls"]:
                        assistant_message["content"].append({
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": json.loads(tool_call["function"]["arguments"])
                        })

                self.conversation_history.append(assistant_message)

                # Add tool results if any
                if tool_results:
                    # For OpenAI models, we'll let unified_llm_completion handle the conversion
                    # For Anthropic models, use the standard format
                    self.conversation_history.append({
                        "role": "user",
                        "content": tool_results
                    })

                # Check if we should continue or if this is the final answer
                if not tool_results:
                    # No tools were called, treat as final answer
                    self.context.log_react_event("react_cycle_complete", {
                        "final_iteration": iteration,
                        "completion_reason": "no_tools_called"
                    })
                    return assistant_content

                self.context.log_react_event("model_response", {
                    "iteration": iteration,
                    "response_length": len(assistant_content),
                    "tools_called": len(tool_results),
                    "response_preview": assistant_content[:200] + "..." if len(assistant_content) > 200 else assistant_content,
                    "input_tokens": response["usage"]["input_tokens"],
                    "output_tokens": response["usage"]["output_tokens"],
                    "total_tokens": response["usage"]["input_tokens"] + response["usage"]["output_tokens"]
                })

            except Exception as e:
                self.context.log_react_event("react_cycle_error", {
                    "iteration": iteration,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                return f"Error in ReAct cycle: {str(e)}"

        # Max iterations reached
        self.context.log_react_event("react_cycle_complete", {
            "final_iteration": max_iterations,
            "completion_reason": "max_iterations_reached"
        })

        return "Maximum iterations reached. The agent was unable to complete the task within the allowed number of steps."


def unified_llm_completion(
    model: str,
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
    tools: Optional[List[Dict]] = None,
    max_tokens: int = 4000,
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Unified LLM interface supporting both Anthropic and OpenAI models via LiteLLM.

    Args:
        model: Model identifier (e.g., "anthropic:claude-sonnet-4-20250514", "openai:gpt-4o")
        messages: List of message dictionaries
        system: System prompt (optional)
        tools: Tool definitions in Anthropic format
        max_tokens: Maximum response tokens
        temperature: Response temperature

    Returns:
        Normalized response dictionary with tool calls and usage information
    """
    # Prepare LiteLLM request
    litellm_messages = []

    # Add system message if provided
    if system:
        litellm_messages.append({"role": "system", "content": system})

    # Add conversation messages with format conversion for OpenAI
    for msg in messages:
        if model.startswith("openai:"):
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                # Convert Anthropic assistant format to OpenAI format
                converted_content = ""
                tool_calls = []

                for content_block in msg["content"]:
                    if content_block.get("type") == "tool_use":
                        # Convert Anthropic tool_use to OpenAI tool_calls
                        tool_calls.append({
                            "id": content_block["id"],
                            "type": "function",
                            "function": {
                                "name": content_block["name"],
                                "arguments": json.dumps(content_block["input"])
                            }
                        })
                    elif content_block.get("type") == "text":
                        converted_content += content_block["text"]

                # Create OpenAI assistant message
                assistant_msg = {
                    "role": "assistant",
                    "content": converted_content if converted_content else None
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls

                litellm_messages.append(assistant_msg)

            elif msg.get("role") == "user" and isinstance(msg.get("content"), list):
                # Convert Anthropic tool result format to OpenAI format
                converted_content = ""
                tool_results = []

                for content_block in msg["content"]:
                    if content_block.get("type") == "tool_result":
                        # For OpenAI, tool results go in a separate "tool" role message
                        tool_results.append({
                            "role": "tool",
                            "content": content_block["content"],
                            "tool_call_id": content_block["tool_use_id"]
                        })
                    elif content_block.get("type") == "text":
                        converted_content += content_block["text"]

                # Add user message if it has text content
                if converted_content.strip():
                    litellm_messages.append({
                        "role": "user",
                        "content": converted_content
                    })

                # Add tool result messages
                litellm_messages.extend(tool_results)

                # Skip adding the original message to avoid duplication
                continue
            else:
                # For simple text messages, add as-is
                litellm_messages.append(msg)
        else:
            # For Anthropic or simple text messages, add as-is
            litellm_messages.append(msg)

    # Convert model name to LiteLLM format and handle tools
    if model.startswith("anthropic:"):
        # For Anthropic: Use claude-3-5-haiku-20241022 instead of anthropic:claude-3-5-haiku-20241022
        litellm_model = model.replace("anthropic:", "")
        litellm_tools = tools  # Keep Anthropic format
    elif model.startswith("openai:"):
        # For OpenAI: Use gpt-4o instead of openai:gpt-4o
        litellm_model = model.replace("openai:", "")
        # Convert tools to OpenAI format if needed
        litellm_tools = None
        if tools:
            litellm_tools = []
            for tool in tools:
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"]
                    }
                }
                litellm_tools.append(openai_tool)
    else:
        # Fallback for unknown provider
        litellm_model = model
        litellm_tools = tools

    # Make LiteLLM call
    response = litellm.completion(
        model=litellm_model,
        messages=litellm_messages,
        tools=litellm_tools,
        max_tokens=max_tokens,
        temperature=temperature
    )

    # Normalize response format with robust token tracking
    # LiteLLM standardizes to input_tokens and output_tokens for all providers
    normalized_response = {
        "content": [],
        "usage": {
            "input_tokens": getattr(response.usage, "input_tokens",
                                   getattr(response.usage, "prompt_tokens", 0)),
            "output_tokens": getattr(response.usage, "output_tokens",
                                    getattr(response.usage, "completion_tokens", 0))
        }
    }

    # Handle text content
    message_content = response.choices[0].message.content
    if message_content:
        normalized_response["content"].append({
            "type": "text",
            "text": message_content
        })

    # Handle tool calls (normalize between providers)
    tool_calls = getattr(response.choices[0].message, "tool_calls", None)
    if tool_calls:
        for tool_call in tool_calls:
            if model.startswith("openai:"):
                # OpenAI format - convert to Anthropic format
                normalized_response["content"].append({
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": json.loads(tool_call.function.arguments)
                })
            else:
                # Anthropic format via LiteLLM - convert properly
                if hasattr(tool_call, 'function'):
                    # LiteLLM may wrap Anthropic responses in OpenAI format
                    normalized_response["content"].append({
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "input": json.loads(tool_call.function.arguments)
                    })
                else:
                    # Direct Anthropic format
                    normalized_response["content"].append({
                        "type": "tool_use",
                        "id": getattr(tool_call, 'id', ''),
                        "name": getattr(tool_call, 'name', ''),
                        "input": getattr(tool_call, 'input', {})
                    })

    return normalized_response


class NativeTransparentAgent:
    """Crystal-clear agent using native Anthropic tool calling"""

    def __init__(self, model: str, max_iterations: int = 10, analyst_max_tokens: int = 16000):
        self.model = model
        self.max_iterations = max_iterations
        self.analyst_max_tokens = analyst_max_tokens

    def execute_query(self, context: AgentContext, system_prompt: str) -> str:
        """Main execution loop with native tool calling"""

        context.log_react_event("agent_execution_start", {
            "model": self.model,
            "max_iterations": self.max_iterations,
            "query": context.query
        })

        # Create ReAct executor with native tool calling
        react_executor = NativeReActExecutor(context, self.model, self.analyst_max_tokens)

        try:
            # Execute ReAct cycle
            result = react_executor.execute_react_cycle(
                system_prompt,
                context.query,
                self.max_iterations
            )

            context.log_react_event("agent_execution_complete", {
                "success": True,
                "result_length": len(result)
            })

            return result

        except Exception as e:
            context.log_react_event("agent_execution_error", {
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise


class ExplicitReActExecutor:
    """Formal ReAct loop with explicit reasoning and automatic critic evaluation"""

    def __init__(self, context: AgentContext, analyst_model: str,
                 critic_model: str, enable_critic: bool = True,
                 analyst_max_tokens: int = 16000, critic_max_tokens: int = 16000):
        self.context = context
        self.analyst_model = analyst_model
        self.critic_model = critic_model
        self.enable_critic = enable_critic
        self.analyst_max_tokens = analyst_max_tokens
        self.critic_max_tokens = critic_max_tokens
        self.anthropic_client = anthropic.Anthropic()
        self.tool_executor = NativeToolExecutor(context)

        # ReAct conversation state
        self.conversation_history = []
        self.react_steps = []

        # Tool definitions - streamlined for efficiency
        self.tools = [
            {
                "name": "write_file_and_run_python",
                "description": "Write a Python script to a file and execute it in the workspace. CRITICAL: Both file_path AND content parameters are MANDATORY - you must provide the complete Python code in the content parameter.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the Python file to create (e.g., 'analysis.py') - REQUIRED"
                        },
                        "content": {
                            "type": "string",
                            "description": "Complete Python code content to write and execute - REQUIRED. Never omit this parameter. Must contain the full script code."
                        }
                    },
                    "required": ["file_path", "content"]
                }
            },
            {
                "name": "stop",
                "description": "Stop the ReAct loop and provide final answer",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "final_answer": {
                            "type": "string",
                            "description": "Final answer or conclusion to the query"
                        }
                    },
                    "required": ["final_answer"]
                }
            }
        ]

    def _call_critic(self, step_data: Dict[str, Any], original_query: str = None, previous_critic_feedback: List[Dict] = None) -> Dict[str, Any]:
        """Call OpenAI critic to evaluate the current step with full context"""
        try:
            # Console logging before critic LLM call
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            iteration = step_data.get('iteration', '?')
            react_stage = f"ReAct Iteration {iteration} - Critic Evaluation"
            if self.context.console_updates_enabled:
                print(f"[{timestamp}] 🧠 LLM CALL: {react_stage} | Model: {self.critic_model}")

            # Extract code and results if action was code execution
            executed_code = ""
            execution_results = ""
            if step_data.get('action_type') == 'write_file_and_run_python':
                action_params = step_data.get('action_params', {})
                executed_code = action_params.get('content', '')
                execution_results = step_data.get('observation', '')

            # Build context from previous critic feedback
            previous_feedback_context = ""
            if previous_critic_feedback:
                previous_feedback_context = "\n\nPREVIOUS CRITIC FEEDBACK:\n"
                for i, feedback in enumerate(previous_critic_feedback[-2:]):  # Last 2 feedbacks
                    previous_feedback_context += f"Step {i}: Issues: {feedback.get('issues', [])} | Suggestions: {feedback.get('suggestions', [])}\n"

            critic_prompt = f"""You are a world-class expert in causal inference methodologies and quantitative trading strategies, evaluating a ReAct (Reasoning-Action-Observation) step in a data analysis workflow.

EXPERTISE AREAS:
- Causal inference (Difference-in-Differences, Instrumental Variables, Regression Discontinuity, Event Studies)
- Time series econometrics and interrupted time series analysis
- Natural gas and energy markets trading strategies
- Pipeline flow analysis and market impact assessment
- Statistical methodology and experimental design

ORIGINAL RESEARCH QUESTION: {original_query or 'Not provided'}

CURRENT STEP ANALYSIS:
THOUGHT: {step_data.get('thought', 'N/A')}
ACTION: {step_data.get('action_type', 'N/A')} with parameters: {step_data.get('action_params', {})}

EXECUTED CODE (if applicable):
```python
{executed_code if executed_code else 'No code executed'}
```

EXECUTION RESULTS:
{execution_results if execution_results else 'No results available'}

{previous_feedback_context}

CAUSAL INFERENCE ASSESSMENT:
As an expert in causal inference and trading strategies, evaluate whether this step represents an appropriately designed causal analysis. Consider:

1. IDENTIFICATION STRATEGY: Does the approach establish causal identification?
2. CONFOUNDING CONTROLS: Are potential confounders addressed?
3. TEMPORAL ANALYSIS: Is the timing of treatment/event properly handled?
4. STATISTICAL RIGOR: Are appropriate econometric methods used?
5. TRADING RELEVANCE: Does this provide actionable market intelligence?
6. DATA VALIDITY: Is the data appropriate for causal claims?

Provide detailed feedback in this JSON format:
{{
    "quality_score": <1-10 integer>,
    "causal_inference_rigor": <1-10 integer>,
    "trading_strategy_relevance": <1-10 integer>,
    "statistical_methodology": <1-10 integer>,
    "data_appropriateness": <1-10 integer>,
    "issues": ["list", "of", "specific", "methodological", "issues"],
    "suggestions": ["detailed", "improvement", "recommendations"],
    "improved_code": "REQUIRED: Provide specific, executable Python code that addresses identified issues. Include complete working examples, not just descriptions. Focus on practical implementation that can be executed immediately.",
    "allow_finish": <true/false>,
    "causal_assessment": "Brief assessment of whether this constitutes proper causal analysis"
}}

CRITICAL FOCUS: This should be a rigorous causal inference analysis, not just descriptive statistics. Evaluate accordingly.

IMPORTANT: Respond with ONLY valid JSON. Do not include any text before or after the JSON object. Ensure all strings are properly escaped and the JSON is valid."""

            response = unified_llm_completion(
                model=self.critic_model,
                messages=[
                    {"role": "user", "content": critic_prompt}
                ],
                system="You are a critical evaluator providing structured feedback.",
                temperature=0.3,
                max_tokens=self.critic_max_tokens
            )

            # Extract text content from normalized response format
            feedback_text = ""
            for content_block in response["content"]:
                if content_block["type"] == "text":
                    feedback_text += content_block["text"]

            # Extract token usage from critic response
            critic_input_tokens = response.get("usage", {}).get("input_tokens", 0)
            critic_output_tokens = response.get("usage", {}).get("output_tokens", 0)
            critic_total_cost = calculate_token_cost(self.critic_model, critic_input_tokens, critic_output_tokens)

            # Try to parse JSON, fallback to text if parsing fails
            try:
                import json
                # Handle case where feedback_text might still be a list
                if isinstance(feedback_text, list):
                    # Convert list to string if needed
                    feedback_text_str = str(feedback_text[0]) if feedback_text else ""
                else:
                    feedback_text_str = feedback_text

                feedback = json.loads(feedback_text_str)
            except json.JSONDecodeError as e:
                # Save the raw response for debugging and preserve all feedback
                import time
                timestamp = int(time.time())
                raw_response_file = f"{self.context.workspace_dir}/critic_raw_response_{timestamp}.txt"
                with open(raw_response_file, 'w') as f:
                    f.write(str(feedback_text))

                # Use the raw text as suggestions - this preserves all the critic's feedback
                # The analyst will get the complete response even if JSON parsing fails
                feedback = {
                    "quality_score": 5,
                    "causal_inference_rigor": 5,
                    "trading_strategy_relevance": 5,
                    "statistical_methodology": 5,
                    "data_appropriateness": 5,
                    "issues": [f"JSON parsing failed: {str(e)} - Raw response saved to {raw_response_file}"],
                    "suggestions": [f"Raw critic response (JSON parsing failed): {feedback_text}"],
                    "improved_code": "",  # Will be extracted by analyst from suggestions
                    "allow_finish": False,
                    "causal_assessment": "Raw critic response preserved - analyst should extract insights manually"
                }

            # Add token usage data to feedback
            feedback["token_usage"] = {
                "input_tokens": critic_input_tokens,
                "output_tokens": critic_output_tokens,
                "total_cost": critic_total_cost,
                "model": self.critic_model
            }

            return feedback

        except Exception as e:
            self.context.console_update(f"Critic evaluation failed: {str(e)}")
            return {
                "quality_score": 5,
                "causal_inference_rigor": 5,
                "trading_strategy_relevance": 5,
                "statistical_methodology": 5,
                "data_appropriateness": 5,
                "issues": [f"Critic error: {str(e)}"],
                "suggestions": ["Continue without critic feedback"],
                "improved_code": "",
                "allow_finish": False,
                "causal_assessment": f"Unable to assess due to error: {str(e)}",
                "token_usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_cost": 0.0,
                    "model": self.critic_model,
                    "error": str(e)
                }
            }

    def _handle_stop_action(self, tool_input: dict, assistant_content: str, iteration: int, user_query: str) -> tuple[bool, str]:
        """Handle stop action with critic approval check. Returns (should_stop, final_answer)"""
        final_answer = tool_input.get("final_answer", assistant_content)

        # If critic is enabled, check if finishing is allowed
        if self.enable_critic:
            self.context.console_update(f"Analyst wants to finish. Checking with expert critic...")

            # Create step data for critic evaluation
            stop_step_data = {
                "iteration": iteration,
                "thought": assistant_content,
                "action_type": "stop",
                "action_params": tool_input,
                "observation": f"Agent attempting to stop with final answer: {final_answer[:200]}..."
            }

            critic_feedback = self._call_critic(stop_step_data, user_query, self.previous_critic_feedback)
            self.previous_critic_feedback.append(critic_feedback)

            # Log critic decision on finishing
            self.context.log_react_event("critic_evaluation", {
                "iteration": iteration,
                "quality_score": critic_feedback.get("quality_score", 0),
                "causal_inference_rigor": critic_feedback.get("causal_inference_rigor", 0),
                "trading_strategy_relevance": critic_feedback.get("trading_strategy_relevance", 0),
                "statistical_methodology": critic_feedback.get("statistical_methodology", 0),
                "data_appropriateness": critic_feedback.get("data_appropriateness", 0),
                "allow_finish": critic_feedback.get("allow_finish", False),
                "issues": critic_feedback.get("issues", []),
                "suggestions": critic_feedback.get("suggestions", []),
                "causal_assessment": critic_feedback.get("causal_assessment", ""),
                "improved_code": critic_feedback.get("improved_code", ""),
                "has_improved_code": bool(critic_feedback.get("improved_code", "").strip()),
                "input_tokens": critic_feedback.get("token_usage", {}).get("input_tokens", 0),
                "output_tokens": critic_feedback.get("token_usage", {}).get("output_tokens", 0),
                "total_cost": critic_feedback.get("token_usage", {}).get("total_cost", 0.0),
                "finish_blocked": not critic_feedback.get("allow_finish", False)
            })

            if critic_feedback.get("allow_finish", False):
                self.context.console_update("✅ Critic approved finishing the analysis")
                self.context.log_react_event("explicit_react_cycle_complete", {
                    "final_iteration": iteration,
                    "completion_reason": "stop_action_approved_by_critic",
                    "final_answer": final_answer
                })
                return True, final_answer
            else:
                self.context.console_update("❌ Critic blocked finishing - analysis needs improvement")
                self._add_critic_feedback_to_conversation(critic_feedback)
                return False, ""
        else:
            # No critic - allow immediate stop
            self.context.log_react_event("explicit_react_cycle_complete", {
                "final_iteration": iteration,
                "completion_reason": "stop_action_called_no_critic",
                "final_answer": final_answer
            })
            return True, final_answer

    def _add_critic_feedback_to_conversation(self, critic_feedback: dict):
        """Add critic feedback to conversation history for next iteration"""
        critic_summary = f"""
EXPERT CRITIC FEEDBACK - ANALYSIS NOT YET COMPLETE:

Quality Issues Identified:
{chr(10).join(f"• {issue}" for issue in critic_feedback.get("issues", []))}

Required Improvements:
{chr(10).join(f"• {suggestion}" for suggestion in critic_feedback.get("suggestions", []))}

Causal Inference Assessment: {critic_feedback.get("causal_assessment", "See above issues")}

You must address these methodological concerns before providing a final answer. Continue with the next iteration to implement the recommended improvements."""

        # Add critic feedback to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": f"I want to provide my final answer, but I need to address some methodological concerns first."
        })
        self.conversation_history.append({
            "role": "user",
            "content": critic_summary
        })

    def _process_assistant_response(self, response, iteration: int, user_query: str) -> tuple[str, list, dict, bool]:
        """Process assistant response and extract content, tools, step data. Returns (assistant_content, tool_results, step_data, should_continue)"""
        assistant_content = ""
        tool_results = []
        step_data = {"iteration": iteration}

        # Process response content - extract thought and action
        # Get text content
        if "content" in response and response["content"]:
            assistant_content = response["content"]
            step_data["thought"] = response["content"]

        # Process tool calls
        if "tool_calls" in response and response["tool_calls"]:
            tool_call = response["tool_calls"][0]  # Get first tool call
            tool_name = tool_call["function"]["name"]
            tool_input = json.loads(tool_call["function"]["arguments"])
            tool_use_id = tool_call["id"]

            step_data["action_type"] = tool_name
            step_data["action_params"] = tool_input

            self.context.log_react_event("explicit_tool_execution_start", {
                "iteration": iteration,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_use_id": tool_use_id
            })

            # Special handling for 'stop' action - need to check critic first
            if tool_name == "stop":
                should_stop, final_answer = self._handle_stop_action(tool_input, assistant_content, iteration, user_query)
                if should_stop:
                    return final_answer, tool_results, step_data, False  # should_continue = False
                else:
                    return assistant_content, tool_results, step_data, True  # Continue to next iteration

            # Execute the tool
            tool_result = self.tool_executor.execute_tool(tool_name, tool_input)
            step_data["observation"] = tool_result.content

            self.context.log_react_event("explicit_tool_execution_complete", {
                "iteration": iteration,
                "tool_name": tool_name,
                "success": tool_result.success,
                "result_length": len(tool_result.content),
                "duration": tool_result.duration,
                "input_tokens": tool_result.input_tokens or 0,
                "output_tokens": tool_result.output_tokens or 0,
                "total_cost": tool_result.total_cost or 0.0
            })

            # Add tool result to conversation
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": tool_result.content,
                "is_error": not tool_result.success
            })

        return assistant_content, tool_results, step_data, True

    def _evaluate_with_critic(self, step_data: dict, iteration: int, user_query: str, tool_results: list):
        """Run critic evaluation if enabled and we have tool actions"""
        if self.enable_critic and tool_results:
            self.context.console_update(f"Running expert causal inference critic for iteration {iteration}...")
            critic_feedback = self._call_critic(step_data, user_query, self.previous_critic_feedback)
            step_data["critic_feedback"] = critic_feedback
            self.previous_critic_feedback.append(critic_feedback)

            self.context.log_react_event("critic_evaluation", {
                "iteration": iteration,
                "quality_score": critic_feedback.get("quality_score", 0),
                "causal_inference_rigor": critic_feedback.get("causal_inference_rigor", 0),
                "trading_strategy_relevance": critic_feedback.get("trading_strategy_relevance", 0),
                "statistical_methodology": critic_feedback.get("statistical_methodology", 0),
                "data_appropriateness": critic_feedback.get("data_appropriateness", 0),
                "allow_finish": critic_feedback.get("allow_finish", False),
                "issues": critic_feedback.get("issues", []),
                "suggestions": critic_feedback.get("suggestions", []),
                "causal_assessment": critic_feedback.get("causal_assessment", ""),
                "improved_code": critic_feedback.get("improved_code", ""),
                "has_improved_code": bool(critic_feedback.get("improved_code", "").strip()),
                "input_tokens": critic_feedback.get("token_usage", {}).get("input_tokens", 0),
                "output_tokens": critic_feedback.get("token_usage", {}).get("output_tokens", 0),
                "total_cost": critic_feedback.get("token_usage", {}).get("total_cost", 0.0)
            })

    def _critic_blocks_finishing(self) -> bool:
        """Check if critic blocks finishing based on latest feedback"""
        if not self.enable_critic or not self.previous_critic_feedback:
            return False

        last_feedback = self.previous_critic_feedback[-1]
        return not last_feedback.get("allow_finish", False)

    def _critic_allows_finishing(self) -> bool:
        """Check if critic allows finishing based on latest feedback"""
        if not self.enable_critic:
            return True  # No critic means always allowed

        if not self.previous_critic_feedback:
            return False  # No feedback yet means not allowed

        last_feedback = self.previous_critic_feedback[-1]
        return last_feedback.get("allow_finish", False)

    def _critic_blocks_stop_action(self, user_query: str, assistant_content: str, tool_input: dict) -> bool:
        """Evaluate stop action with critic and return whether it's blocked"""
        if not self.enable_critic:
            return False

        self.context.console_update("Analyst wants to finish. Checking with expert critic...")

        # Create step data for critic evaluation
        final_answer = tool_input.get("final_answer", assistant_content)
        stop_step_data = {
            "iteration": len(self.react_steps),
            "thought": assistant_content,
            "action_type": "stop",
            "action_params": tool_input,
            "observation": f"Agent attempting to stop with final answer: {final_answer[:200]}..."
        }

        # Get critic feedback
        critic_feedback = self._call_critic(stop_step_data, user_query, self.previous_critic_feedback)
        self.previous_critic_feedback.append(critic_feedback)

        # Log critic decision
        self.context.log_react_event("critic_evaluation", {
            "iteration": len(self.react_steps),
            "quality_score": critic_feedback.get("quality_score", 0),
            "causal_inference_rigor": critic_feedback.get("causal_inference_rigor", 0),
            "trading_strategy_relevance": critic_feedback.get("trading_strategy_relevance", 0),
            "statistical_methodology": critic_feedback.get("statistical_methodology", 0),
            "data_appropriateness": critic_feedback.get("data_appropriateness", 0),
            "allow_finish": critic_feedback.get("allow_finish", False),
            "issues": critic_feedback.get("issues", []),
            "suggestions": critic_feedback.get("suggestions", []),
            "causal_assessment": critic_feedback.get("causal_assessment", ""),
            "improved_code": critic_feedback.get("improved_code", ""),
            "has_improved_code": bool(critic_feedback.get("improved_code", "").strip()),
            "input_tokens": critic_feedback.get("token_usage", {}).get("input_tokens", 0),
            "output_tokens": critic_feedback.get("token_usage", {}).get("output_tokens", 0),
            "total_cost": critic_feedback.get("token_usage", {}).get("total_cost", 0.0),
            "finish_blocked": not critic_feedback.get("allow_finish", False)
        })

        allowed = critic_feedback.get("allow_finish", False)
        if allowed:
            self.context.console_update("✅ Critic approved finishing the analysis")

        return not allowed

    def _add_critic_guidance_to_conversation(self):
        """Add critic feedback to conversation history to guide next iteration"""
        if not self.previous_critic_feedback:
            return

        last_feedback = self.previous_critic_feedback[-1]

        # Create concise guidance message
        guidance = "The expert critic has said we are not yet ready to finish this analysis. "

        if last_feedback.get("suggestions"):
            guidance += "Please incorporate their suggestions: "
            guidance += "; ".join(last_feedback.get("suggestions", []))

        if last_feedback.get("improved_code") and last_feedback.get("improved_code").strip():
            guidance += "\n\nThe critic also provided improved code you should consider implementing."

        self.conversation_history.append({
            "role": "user",
            "content": guidance
        })

    def _get_analyst_response(self, system_prompt: str) -> Any:
        """Get response from analyst model using unified LLM interface"""
        # Console logging before analyst LLM call
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        iteration = len(self.react_steps)
        react_stage = f"ReAct Iteration {iteration + 1} - Analyst Reasoning"
        if self.context.console_updates_enabled:
            print(f"[{timestamp}] 🧠 LLM CALL: {react_stage} | Model: {self.analyst_model}")

        return unified_llm_completion(
            model=self.analyst_model,
            messages=self.conversation_history,
            system=system_prompt,
            tools=self.tools,
            max_tokens=self.analyst_max_tokens,
            temperature=0.0
        )

    def _process_response_content(self, response: Any, iteration: int) -> tuple[str, list, dict]:
        """Process response content and extract thought, actions, and step data"""
        assistant_content = ""
        tool_results = []
        step_data = {"iteration": iteration}

        # Extract thought and actions from response (normalized format)
        for content_block in response["content"]:
            if content_block["type"] == "text":
                assistant_content += content_block["text"]
                step_data["thought"] = content_block["text"]
            elif content_block["type"] == "tool_use":
                tool_name = content_block["name"]
                tool_input = content_block["input"]
                tool_use_id = content_block["id"]

                step_data["action_type"] = tool_name
                step_data["action_params"] = tool_input

                self.context.log_react_event("explicit_tool_execution_start", {
                    "iteration": iteration,
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "tool_use_id": tool_use_id
                })

                # Process tool action
                tool_result = self._process_tool_action(tool_name, tool_input, assistant_content, iteration)
                if tool_result is None:  # Stop action was handled
                    return assistant_content, [], step_data

                # Add successful tool execution to results
                step_data["observation"] = tool_result.content
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": tool_result.content,
                    "is_error": not tool_result.success
                })

                self.context.log_react_event("explicit_tool_execution_complete", {
                    "iteration": iteration,
                    "tool_name": tool_name,
                    "success": tool_result.success,
                    "result_length": len(tool_result.content),
                    "duration": tool_result.duration,
                    "input_tokens": tool_result.input_tokens or 0,
                    "output_tokens": tool_result.output_tokens or 0,
                    "total_cost": tool_result.total_cost or 0.0
                })

        return assistant_content, tool_results, step_data

    def _process_tool_action(self, tool_name: str, tool_input: dict, assistant_content: str, iteration: int):
        """Process a tool action and return result or None if stop was handled"""
        if tool_name == "stop":
            return self._handle_stop_tool(tool_input, assistant_content, iteration)
        else:
            return self.tool_executor.execute_tool(tool_name, tool_input)

    def _handle_stop_tool(self, tool_input: dict, assistant_content: str, iteration: int):
        """Handle stop tool action, return None if stop is processed"""
        final_answer = tool_input.get("final_answer", assistant_content)

        # Check if critic allows finishing (based on previous feedback)
        if self.enable_critic and not self._critic_allows_finishing():
            self.context.console_update("❌ Stop action blocked - critic has not approved finishing yet")
            # Add guidance to conversation and signal to continue
            self.conversation_history.extend([
                {"role": "assistant", "content": "I want to stop and provide my final answer."},
                {"role": "user", "content": "The expert critic has not yet approved finishing this analysis. Please continue working to address previous critic feedback before attempting to stop."}
            ])
            return None  # Signal that stop was handled and we should continue

        # Stop allowed - log completion and raise special exception to exit cleanly
        self.context.log_react_event("explicit_react_cycle_complete", {
            "final_iteration": iteration,
            "completion_reason": "stop_action_approved",
            "final_answer": final_answer
        })

        # Use a custom exception to cleanly exit the loop with final answer
        raise self.StopApproved(final_answer)

    class StopApproved(Exception):
        """Exception raised when stop action is approved by critic"""
        def __init__(self, final_answer):
            self.final_answer = final_answer

    def _build_assistant_message(self, response: Any, assistant_content: str) -> Dict[str, Any]:
        """Build assistant message from response content"""
        if "content" in response and response["content"]:
            # Anthropic format - use normalized response content directly
            return {"role": "assistant", "content": response["content"]}

        # Fallback format construction
        content = []
        if assistant_content:
            content.append({"type": "text", "text": assistant_content})

        # Add tool uses for OpenAI format
        if "tool_calls" in response and response["tool_calls"]:
            content.extend(self._convert_tool_calls_to_content(response["tool_calls"]))

        return {"role": "assistant", "content": content}

    def _convert_tool_calls_to_content(self, tool_calls: list) -> list:
        """Convert OpenAI tool calls to Anthropic content format"""
        return [
            {
                "type": "tool_use",
                "id": tool_call["id"],
                "name": tool_call["function"]["name"],
                "input": json.loads(tool_call["function"]["arguments"])
            }
            for tool_call in tool_calls
        ]

    def _build_tool_result_message(self, tool_results: list) -> Dict[str, Any]:
        """Build user message with tool results for unified_llm_completion handling"""
        tool_result_content = [
            {
                "type": "tool_result",
                "tool_use_id": tool_result["tool_use_id"],
                "content": tool_result["content"],
                "is_error": tool_result.get("is_error", False)
            }
            for tool_result in tool_results
        ]

        return {"role": "user", "content": tool_result_content}

    def _add_messages_to_conversation(self, response: Any, assistant_content: str, tool_results: list):
        """Add assistant message and tool results to conversation history"""
        # Add assistant message
        assistant_message = self._build_assistant_message(response, assistant_content)
        self.conversation_history.append(assistant_message)

        # Add tool results if any
        if tool_results:
            tool_result_message = self._build_tool_result_message(tool_results)
            self.conversation_history.append(tool_result_message)

    def _run_critic_evaluation(self, step_data: dict, user_query: str, iteration: int):
        """Run critic evaluation and add feedback to conversation"""
        if not self.enable_critic:
            return

        self.context.console_update(f"Running expert causal inference critic for iteration {iteration}...")
        critic_feedback = self._call_critic(step_data, user_query, self.previous_critic_feedback)
        step_data["critic_feedback"] = critic_feedback
        self.previous_critic_feedback.append(critic_feedback)

        # Log critic evaluation
        self.context.log_react_event("critic_evaluation", {
            "iteration": iteration,
            "quality_score": critic_feedback.get("quality_score", 0),
            "causal_inference_rigor": critic_feedback.get("causal_inference_rigor", 0),
            "trading_strategy_relevance": critic_feedback.get("trading_strategy_relevance", 0),
            "statistical_methodology": critic_feedback.get("statistical_methodology", 0),
            "data_appropriateness": critic_feedback.get("data_appropriateness", 0),
            "allow_finish": critic_feedback.get("allow_finish", False),
            "issues": critic_feedback.get("issues", []),
            "suggestions": critic_feedback.get("suggestions", []),
            "causal_assessment": critic_feedback.get("causal_assessment", ""),
            "improved_code": critic_feedback.get("improved_code", ""),
            "has_improved_code": bool(critic_feedback.get("improved_code", "").strip()),
            "input_tokens": critic_feedback.get("token_usage", {}).get("input_tokens", 0),
            "output_tokens": critic_feedback.get("token_usage", {}).get("output_tokens", 0),
            "total_cost": critic_feedback.get("token_usage", {}).get("total_cost", 0.0)
        })

        # Add critic feedback to conversation
        self._add_critic_feedback_to_conversation(critic_feedback)

    def _add_critic_feedback_to_conversation(self, critic_feedback: dict):
        """Format and add critic feedback to conversation history"""
        critic_message = f"\n\n**EXPERT CAUSAL INFERENCE & TRADING STRATEGY CRITIC FEEDBACK:**\n"
        critic_message += f"Overall Quality: {critic_feedback.get('quality_score', 'N/A')}/10\n"
        critic_message += f"Causal Inference Rigor: {critic_feedback.get('causal_inference_rigor', 'N/A')}/10\n"
        critic_message += f"Trading Strategy Relevance: {critic_feedback.get('trading_strategy_relevance', 'N/A')}/10\n"
        critic_message += f"Statistical Methodology: {critic_feedback.get('statistical_methodology', 'N/A')}/10\n"
        critic_message += f"Data Appropriateness: {critic_feedback.get('data_appropriateness', 'N/A')}/10\n\n"

        critic_message += f"**Causal Assessment:** {critic_feedback.get('causal_assessment', 'Not provided')}\n\n"

        # Critical information: whether finishing is allowed
        if critic_feedback.get("allow_finish", False):
            critic_message += "✅ **FINISHING APPROVED**: You may use the 'stop' tool in your next response if you have completed the analysis.\n\n"
        else:
            critic_message += "❌ **CONTINUE REQUIRED**: You must continue improving the analysis before stopping.\n\n"

        if critic_feedback.get('issues'):
            critic_message += f"**Critical Issues:** {'; '.join(critic_feedback.get('issues', []))}\n\n"

        if critic_feedback.get('suggestions'):
            critic_message += f"**Expert Recommendations:** {'; '.join(critic_feedback.get('suggestions', []))}\n\n"

        if critic_feedback.get('improved_code') and critic_feedback.get('improved_code').strip():
            critic_message += f"**Improved Code Suggestion:**\n```python\n{critic_feedback.get('improved_code')}\n```\n"

        # Add critic feedback as a separate user message to guide next iteration
        self.conversation_history.append({
            "role": "user",
            "content": critic_message
        })

    def _log_model_response(self, response: Any, iteration: int, assistant_content: str, tool_results: list):
        """Log model response with token usage"""
        # Handle both normalized dict and object response formats
        if isinstance(response, dict) and "usage" in response:
            input_tokens = response["usage"].get("input_tokens", 0)
            output_tokens = response["usage"].get("output_tokens", 0)
        else:
            input_tokens = getattr(response.usage, 'input_tokens', 0) if hasattr(response, 'usage') else 0
            output_tokens = getattr(response.usage, 'output_tokens', 0) if hasattr(response, 'usage') else 0
        total_cost = calculate_token_cost(self.analyst_model, input_tokens, output_tokens)

        self.context.log_react_event("explicit_model_response", {
            "iteration": iteration,
            "response_length": len(assistant_content),
            "tools_called": len(tool_results),
            "critic_enabled": self.enable_critic,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_cost": total_cost
        })

    def execute_react_cycle(self, system_prompt: str, user_query: str, max_iterations: int = 10) -> str:
        """Execute formal ReAct cycle with explicit reasoning and automatic critic"""

        # Initialize conversation and state
        self.conversation_history = [{"role": "user", "content": user_query}]
        self.previous_critic_feedback = []

        self.context.log_react_event("explicit_react_cycle_start", {
            "max_iterations": max_iterations,
            "analyst_model": self.analyst_model,
            "critic_model": self.critic_model,
            "critic_enabled": self.enable_critic,
            "user_query": user_query
        })

        # Main ReAct loop
        for iteration in range(max_iterations):
            self.context.log_react_event("explicit_react_iteration_start", {
                "iteration": iteration,
                "conversation_length": len(self.conversation_history)
            })

            try:
                # 1. Get analyst response
                response = self._get_analyst_response(system_prompt)

                # 2. Process response content and execute tools
                assistant_content, tool_results, step_data = self._process_response_content(response, iteration)

                # 3. Store step and add messages to conversation
                self.react_steps.append(step_data)
                self._add_messages_to_conversation(response, assistant_content, tool_results)

                # 4. Run critic evaluation if tools were executed
                if tool_results:
                    self._run_critic_evaluation(step_data, user_query, iteration)

                # 5. Check termination conditions
                if not tool_results:
                    # No tools called - natural completion
                    self.context.log_react_event("explicit_react_cycle_complete", {
                        "final_iteration": iteration,
                        "completion_reason": "no_tools_called"
                    })
                    return assistant_content

                # 6. Log model response metrics
                self._log_model_response(response, iteration, assistant_content, tool_results)

            except self.StopApproved as stop_exception:
                # Clean exit when stop is approved
                return stop_exception.final_answer

            except Exception as e:
                self.context.log_react_event("explicit_react_cycle_error", {
                    "iteration": iteration,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                return f"Error in Explicit ReAct cycle: {str(e)}"

        # Max iterations reached
        self.context.log_react_event("explicit_react_cycle_complete", {
            "final_iteration": max_iterations,
            "completion_reason": "max_iterations_reached"
        })

        return "Maximum iterations reached. The agent was unable to complete the task within the allowed number of steps."


class ExplicitReActAgent:
    """Agent using formal ReAct loop with automatic critic evaluation"""

    def __init__(self, analyst_model: str,
                 critic_model: str, enable_critic: bool = True, max_iterations: int = 10,
                 analyst_max_tokens: int = 16000, critic_max_tokens: int = 16000):
        self.analyst_model = analyst_model
        self.critic_model = critic_model
        self.enable_critic = enable_critic
        self.max_iterations = max_iterations
        self.analyst_max_tokens = analyst_max_tokens
        self.critic_max_tokens = critic_max_tokens

    def execute_query(self, context: AgentContext, system_prompt: str) -> str:
        """Main execution loop with explicit ReAct and critic"""

        context.log_react_event("explicit_agent_execution_start", {
            "analyst_model": self.analyst_model,
            "critic_model": self.critic_model,
            "enable_critic": self.enable_critic,
            "max_iterations": self.max_iterations,
            "query": context.query
        })

        # Create explicit ReAct executor
        react_executor = ExplicitReActExecutor(
            context,
            self.analyst_model,
            self.critic_model,
            self.enable_critic,
            self.analyst_max_tokens,
            self.critic_max_tokens
        )

        try:
            # Execute ReAct cycle
            result = react_executor.execute_react_cycle(
                system_prompt,
                context.query,
                self.max_iterations
            )

            context.log_react_event("explicit_agent_execution_complete", {
                "success": True,
                "result_length": len(result),
                "steps_completed": len(react_executor.react_steps)
            })

            return result

        except Exception as e:
            context.log_react_event("explicit_agent_execution_error", {
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise


def create_native_transparent_agent(model_name: str, max_iterations: int = 10, analyst_max_tokens: int = 16000) -> NativeTransparentAgent:
    """Create a native transparent agent with the specified model"""
    return NativeTransparentAgent(model_name, max_iterations, analyst_max_tokens)


def create_explicit_react_agent(analyst_model: str,
                                critic_model: str,
                                enable_critic: bool = True,
                                max_iterations: int = 10,
                                analyst_max_tokens: int = 16000,
                                critic_max_tokens: int = 16000) -> ExplicitReActAgent:
    """Create an explicit ReAct agent with critic evaluation"""
    return ExplicitReActAgent(analyst_model, critic_model, enable_critic, max_iterations,
                             analyst_max_tokens, critic_max_tokens)
