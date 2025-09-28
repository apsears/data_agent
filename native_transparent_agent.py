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
from retry_ledger import RetryLedger, set_current_ledger, log_execution, log_pre_dispatch, log_post_process


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

    @classmethod
    def success_result(cls, content: str, duration: float = None) -> "ToolResult":
        return cls(success=True, content=content, duration=duration)

    @classmethod
    def error_result(cls, content: str, category: ErrorCategory, details: str = None, duration: float = None) -> "ToolResult":
        return cls(success=False, content=content, error_category=category, error_details=details, duration=duration)


@dataclass
class AgentContext:
    """Context for agent execution"""
    workspace_dir: Path
    query: str
    query_id: str
    dataset_description: str
    analysis_type: str
    rubric: Dict[str, Any]
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
        elif tool_name == "read_file":
            if 'file_path' not in tool_input:
                return ToolResult.error_result(
                    "Missing required argument: file_path",
                    ErrorCategory.VALIDATION_ERROR,
                    "read_file requires file_path argument"
                )
        elif tool_name == "list_files":
            pass  # No required arguments
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
        elif tool_name == "read_file":
            return self._execute_read_file(tool_input)
        elif tool_name == "list_files":
            return self._execute_list_files(tool_input)
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

    def _execute_list_files(self, tool_input: dict) -> ToolResult:
        """List files in workspace directory"""
        start_time = time.time()

        try:
            files = []
            for item in self.context.workspace_dir.iterdir():
                if item.is_file():
                    files.append(f"📄 {item.name}")
                elif item.is_dir():
                    files.append(f"📁 {item.name}/")

            duration = time.time() - start_time

            if not files:
                content = "Workspace is empty"
            else:
                content = "Files in workspace:\n" + "\n".join(sorted(files))

            return ToolResult.success_result(content, duration)

        except Exception as e:
            duration = time.time() - start_time
            return ToolResult.error_result(
                f"Error listing files: {str(e)}",
                ErrorCategory.SYSTEM_ERROR,
                str(e),
                duration
            )


class NativeReActExecutor:
    """Native ReAct loop using Anthropic's built-in tool calling"""

    def __init__(self, context: AgentContext, model_name: str):
        self.context = context
        self.model_name = model_name
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
            },
            {
                "name": "read_file",
                "description": "Read the contents of a file in the workspace",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to read"
                        }
                    },
                    "required": ["file_path"]
                }
            },
            {
                "name": "list_files",
                "description": "List all files and directories in the workspace",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]

    def execute_react_cycle(self, system_prompt: str, user_query: str, max_iterations: int = 10) -> str:
        """Execute complete ReAct cycle using native tool calling"""

        self.conversation_history = [{"role": "user", "content": user_query}]

        self.context.log_react_event("react_cycle_start", {
            "max_iterations": max_iterations,
            "system_prompt_length": len(system_prompt),
            "user_query": user_query
        })

        for iteration in range(max_iterations):
            self.context.log_react_event("react_iteration_start", {
                "iteration": iteration,
                "conversation_length": len(self.conversation_history)
            })

            try:
                # Make API call with native tool support
                response = self.client.messages.create(
                    model=self.model_name.replace("anthropic:", ""),
                    max_tokens=8000,  # Increased from 4000 to allow complete tool calls
                    messages=self.conversation_history,
                    system=system_prompt,
                    tools=self.tools
                )

                assistant_content = ""
                tool_results = []

                # Process response content
                for content_block in response.content:
                    if content_block.type == "text":
                        assistant_content += content_block.text
                    elif content_block.type == "tool_use":
                        # Execute tool using native structure
                        tool_name = content_block.name
                        tool_input = content_block.input
                        tool_use_id = content_block.id

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
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        assistant_message["content"].append({
                            "type": "tool_use",
                            "id": content_block.id,
                            "name": content_block.name,
                            "input": content_block.input
                        })

                self.conversation_history.append(assistant_message)

                # Add tool results if any
                if tool_results:
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
                    "response_preview": assistant_content[:200] + "..." if len(assistant_content) > 200 else assistant_content
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


class NativeTransparentAgent:
    """Crystal-clear agent using native Anthropic tool calling"""

    def __init__(self, model: str, max_iterations: int = 10):
        self.model = model
        self.max_iterations = max_iterations

    def execute_query(self, context: AgentContext, system_prompt: str) -> str:
        """Main execution loop with native tool calling"""

        context.log_react_event("agent_execution_start", {
            "model": self.model,
            "max_iterations": self.max_iterations,
            "query": context.query
        })

        # Create ReAct executor with native tool calling
        react_executor = NativeReActExecutor(context, self.model)

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


def create_native_transparent_agent(model_name: str, max_iterations: int = 10) -> NativeTransparentAgent:
    """Create a native transparent agent with the specified model"""
    return NativeTransparentAgent(model_name, max_iterations)