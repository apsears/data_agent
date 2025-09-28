#!/usr/bin/env python3
"""
Transparent Agent Framework - Direct Anthropic API + Custom ReAct Loop

Replacement for PydanticAI with full visibility into every decision point and tool execution.
Following the documented strategy from docs/2025_09_27_19_45_pydantic_ai_removal_strategy.md
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
import re
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
class ToolCall:
    """Structured tool call representation"""
    name: str
    args: Dict[str, Any]
    call_id: Optional[str] = None


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


class ToolExecutor:
    """Direct tool execution with surgical instrumentation"""

    def __init__(self, context: AgentContext):
        self.context = context

    def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute tool with full visibility and instrumentation"""
        start_time = time.time()

        # Phase 1: Pre-dispatch validation
        validation_result = self._validate_tool_call(tool_call)
        if not validation_result.success:
            log_pre_dispatch(tool_call.name, tool_call.args, validation_result.error_details)
            return validation_result

        log_pre_dispatch(tool_call.name, tool_call.args)

        # Phase 2: Execution
        execution_result = self._execute_tool_directly(tool_call)

        # Log execution phase
        log_execution(
            tool_name=tool_call.name,
            exit_code=execution_result.exit_code or (0 if execution_result.success else 1),
            duration_s=execution_result.duration or (time.time() - start_time),
            stderr_path=None,  # Could be enhanced to track stderr files
            stdout_tail=execution_result.content[-500:] if len(execution_result.content) > 500 else execution_result.content,
            stderr_tail=execution_result.error_details[-500:] if execution_result.error_details and len(execution_result.error_details) > 500 else execution_result.error_details
        )

        # Phase 3: Post-process
        log_post_process(tool_call.name)

        return execution_result

    def _validate_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Validate tool call arguments"""
        if tool_call.name == "write_file_and_run_python":
            if 'file_path' not in tool_call.args or 'content' not in tool_call.args:
                return ToolResult.error_result(
                    "Missing required arguments: file_path and content",
                    ErrorCategory.VALIDATION_ERROR,
                    "write_file_and_run_python requires file_path and content arguments"
                )
        elif tool_call.name == "read_file":
            if 'file_path' not in tool_call.args:
                return ToolResult.error_result(
                    "Missing required argument: file_path",
                    ErrorCategory.VALIDATION_ERROR,
                    "read_file requires file_path argument"
                )
        elif tool_call.name == "list_files":
            pass  # No required arguments
        else:
            return ToolResult.error_result(
                f"Unknown tool: {tool_call.name}",
                ErrorCategory.VALIDATION_ERROR,
                f"Tool '{tool_call.name}' is not supported"
            )

        return ToolResult.success_result("Validation passed")

    def _execute_tool_directly(self, tool_call: ToolCall) -> ToolResult:
        """Direct execution without framework abstraction"""
        if tool_call.name == "write_file_and_run_python":
            return self._execute_python_tool(tool_call.args)
        elif tool_call.name == "read_file":
            return self._execute_read_file(tool_call.args)
        elif tool_call.name == "list_files":
            return self._execute_list_files(tool_call.args)
        else:
            return ToolResult.error_result(
                f"Tool execution not implemented: {tool_call.name}",
                ErrorCategory.SYSTEM_ERROR
            )

    def _execute_python_tool(self, args: dict) -> ToolResult:
        """Direct Python execution with full visibility"""
        start_time = time.time()
        file_path = args['file_path']
        content = args['content']

        try:
            # Step 1: Write the file
            full_path = self.context.workspace_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Step 2: Execute the script
            logs_dir = self.context.workspace_dir.parent / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            task_id = f"{int(time.time())}-{file_path.replace('/', '_').replace('.py', '')}"
            stdout_log = (logs_dir / f"stdout-{task_id}.log").resolve()
            stderr_log = (logs_dir / f"stderr-{task_id}.log").resolve()

            # Use uv venv Python - no fallbacks
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
                    [venv_python, file_path],
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

                # Create comprehensive output message
                output = f"Successfully executed {file_path} (exit code: {result.returncode}, duration: {duration:.1f}s)\n"
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

    def _execute_read_file(self, args: dict) -> ToolResult:
        """Read file contents"""
        start_time = time.time()
        file_path = args['file_path']

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

    def _execute_list_files(self, args: dict) -> ToolResult:
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


class ReActExecutor:
    """Transparent ReAct loop with full step visibility"""

    def __init__(self, context: AgentContext, model_name: str):
        self.context = context
        self.model_name = model_name
        self.client = anthropic.Anthropic()
        self.tool_executor = ToolExecutor(context)
        self.conversation_history = []

    def execute_react_cycle(self, system_prompt: str, user_query: str, max_iterations: int = 10) -> str:
        """Execute complete ReAct cycle with comprehensive logging"""

        # Initialize conversation with system message
        self.conversation_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

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

            # Get model response
            try:
                response = self.client.messages.create(
                    model=self.model_name.replace("anthropic:", ""),
                    max_tokens=4000,
                    messages=[msg for msg in self.conversation_history if msg["role"] != "system"],
                    system=system_prompt
                )

                assistant_content = response.content[0].text
                self.conversation_history.append({"role": "assistant", "content": assistant_content})

                self.context.log_react_event("model_response", {
                    "iteration": iteration,
                    "response_length": len(assistant_content),
                    "response_preview": assistant_content[:200] + "..." if len(assistant_content) > 200 else assistant_content
                })

                # Parse for tool calls or final answer
                tool_call = self._parse_tool_call(assistant_content)

                if tool_call:
                    # Execute tool
                    self.context.log_react_event("tool_execution_start", {
                        "iteration": iteration,
                        "tool_name": tool_call.name,
                        "tool_args": tool_call.args
                    })

                    tool_result = self.tool_executor.execute_tool(tool_call)

                    self.context.log_react_event("tool_execution_complete", {
                        "iteration": iteration,
                        "tool_name": tool_call.name,
                        "success": tool_result.success,
                        "result_length": len(tool_result.content),
                        "duration": tool_result.duration
                    })

                    # Add tool result to conversation
                    tool_message = f"Tool execution result:\n{tool_result.content}"
                    self.conversation_history.append({"role": "user", "content": tool_message})

                    if not tool_result.success:
                        self.context.console_update(f"Tool execution failed: {tool_result.content}")
                        # Continue to next iteration to let model handle the error
                        continue

                else:
                    # No tool call found, assume final answer
                    self.context.log_react_event("react_cycle_complete", {
                        "final_iteration": iteration,
                        "completion_reason": "final_answer_detected"
                    })
                    return assistant_content

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

    def _parse_tool_call(self, content: str) -> Optional[ToolCall]:
        """Parse assistant response for tool calls using multiple strategies"""

        # Try different parsing strategies in order of preference
        strategies = [
            self._parse_wrapped_xml_calls,
            self._parse_direct_tool_wrappers,
            self._parse_invoke_blocks,
            self._parse_select_blocks,
            self._parse_function_style_calls
        ]

        for strategy in strategies:
            try:
                result = strategy(content)
                if result:
                    return result
            except Exception as e:
                if hasattr(self, 'context'):
                    self.context.log_react_event("tool_parsing_error", {
                        "strategy": strategy.__name__,
                        "error": str(e)
                    })
                continue

        return None

    def _parse_wrapped_xml_calls(self, content: str) -> Optional[ToolCall]:
        """Parse tool calls wrapped in XML containers like <tool_call> or <anythingllm-function-call>"""

        wrappers = [
            r'<tool_call>(.*?)</tool_call>',
            r'<anythingllm-function-call>(.*?)</anythingllm-function-call>'
        ]

        for pattern in wrappers:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                xml_content = match.group(1).strip()
                tool_call = self._parse_xml_tool_call(xml_content)
                if tool_call:
                    return tool_call

        return None

    def _parse_direct_tool_wrappers(self, content: str) -> Optional[ToolCall]:
        """Parse tool calls with direct tool name wrappers like <write_file_and_run_python>"""

        tool_patterns = {
            'write_file_and_run_python': [
                r'<write_file_and_run_python>(.*?)</write_file_and_run_python>',
                r'<execute>\s*<tool_name>write_file_and_run_python</tool_name>\s*<parameters>(.*?)</parameters>\s*</execute>'
            ],
            'read_file': [
                r'<read_file>(.*?)</read_file>',
                r'<execute>\s*<tool_name>read_file</tool_name>\s*<parameters>(.*?)</parameters>\s*</execute>'
            ],
            'list_files': [
                r'<list_files>(.*?)</list_files>',
                r'<execute>\s*<tool_name>list_files</tool_name>\s*<parameters>(.*?)</parameters>\s*</execute>'
            ]
        }

        for tool_name, patterns in tool_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    xml_content = match.group(1).strip()

                    # Parse based on the format detected
                    if '<tool_name>' in pattern:
                        # New execute format - parse different parameter structure
                        args = self._parse_execute_parameters(xml_content)
                    else:
                        # Original format
                        args = self._parse_parameters_xml(xml_content)

                    tool_call = self._build_tool_call(tool_name, args)
                    if tool_call:
                        return tool_call

        return None

    def _parse_invoke_blocks(self, content: str) -> Optional[ToolCall]:
        """Parse <invoke name="tool_name"> style tool calls"""

        pattern = r'<invoke name="([^"]+)">(.*?)</invoke>'
        match = re.search(pattern, content, re.DOTALL)

        if match:
            tool_name = match.group(1)
            params_xml = match.group(2).strip()
            args = self._parse_parameters_xml(params_xml)

            return self._build_tool_call(tool_name, args)

        return None

    def _parse_select_blocks(self, content: str) -> Optional[ToolCall]:
        """Parse <select>tool_name</select> style tool calls"""

        pattern = r'<select>([^<]+)</select>'
        match = re.search(pattern, content)

        if match:
            tool_name = match.group(1).strip()
            args = self._parse_parameters_from_content(content)

            return self._build_tool_call(tool_name, args)

        return None

    def _build_tool_call(self, tool_name: str, args: Dict[str, str]) -> Optional[ToolCall]:
        """Build a ToolCall object from tool name and arguments"""

        if tool_name == "write_file_and_run_python":
            file_path = args.get("filename") or args.get("file_path")
            content_text = args.get("code") or args.get("content")
            if file_path and content_text:
                return ToolCall(
                    name="write_file_and_run_python",
                    args={"file_path": file_path, "content": content_text}
                )

        elif tool_name == "read_file":
            file_path = args.get("file_path")
            if file_path:
                return ToolCall(name="read_file", args={"file_path": file_path})

        elif tool_name == "list_files":
            return ToolCall(name="list_files", args={})

        return None

    def _parse_xml_tool_call(self, xml_content: str) -> Optional[ToolCall]:
        """Parse XML-formatted tool call"""
        from xml.etree import ElementTree as ET
        try:
            # Wrap in root element if needed
            if not xml_content.startswith('<'):
                return None

            root = ET.fromstring(f"<root>{xml_content}</root>")
            invoke_elem = root.find('invoke')

            if invoke_elem is not None:
                tool_name = invoke_elem.get('name')
                if tool_name:
                    args = {}
                    for param in invoke_elem.findall('parameter'):
                        name = param.get('name')
                        if name and param.text:
                            args[name] = param.text.strip()

                    return ToolCall(name=tool_name, args=args)
        except ET.ParseError:
            return None
        return None

    def _parse_parameters_xml(self, xml_content: str) -> Dict[str, str]:
        """Parse parameter XML elements"""
        args = {}
        try:
            # Look for parameter elements with multiline content support
            param_pattern = r'<parameter name="([^"]+)">(.*?)</parameter>'
            matches = re.findall(param_pattern, xml_content, re.DOTALL)
            for name, value in matches:
                args[name] = value.strip()
        except Exception:
            pass
        return args

    def _parse_execute_parameters(self, xml_content: str) -> Dict[str, str]:
        """Parse parameters from the execute format with script_name and script_content"""
        args = {}
        try:
            # Look for script_name and script_content parameters
            script_name_pattern = r'<script_name>([^<]*)</script_name>'
            script_content_pattern = r'<script_content>(.*?)</script_content>'

            script_name_match = re.search(script_name_pattern, xml_content, re.DOTALL)
            script_content_match = re.search(script_content_pattern, xml_content, re.DOTALL)

            if script_name_match:
                args["filename"] = script_name_match.group(1).strip()
            if script_content_match:
                args["code"] = script_content_match.group(1).strip()

        except Exception:
            pass
        return args

    def _parse_parameters_from_content(self, content: str) -> Dict[str, str]:
        """Parse parameters from content using regex"""
        args = {}
        try:
            # Look for parameter blocks
            param_pattern = r'<parameter name="([^"]+)">\s*(.*?)\s*</parameter>'
            matches = re.findall(param_pattern, content, re.DOTALL)
            for name, value in matches:
                args[name] = value.strip()
        except Exception:
            pass
        return args

    def _parse_function_style_calls(self, content: str) -> Optional[ToolCall]:
        """Fallback to function-style call parsing"""

        # Pattern for write_file_and_run_python function call style
        python_pattern = r'write_file_and_run_python\s*\(\s*file_path\s*=\s*["\']([^"\']+)["\']\s*,\s*content\s*=\s*["\']([^"\']*(?:[^"\'\\]|\\.)*)["\']'
        python_match = re.search(python_pattern, content, re.DOTALL)

        if python_match:
            return ToolCall(
                name="write_file_and_run_python",
                args={
                    "file_path": python_match.group(1),
                    "content": python_match.group(2)
                }
            )

        # Pattern for read_file
        read_pattern = r'read_file\s*\(\s*file_path\s*=\s*["\']([^"\']+)["\']\s*\)'
        read_match = re.search(read_pattern, content)

        if read_match:
            return ToolCall(
                name="read_file",
                args={"file_path": read_match.group(1)}
            )

        # Pattern for list_files
        list_pattern = r'list_files\s*\(\s*\)'
        list_match = re.search(list_pattern, content)

        if list_match:
            return ToolCall(name="list_files", args={})

        return None


class TransparentAgent:
    """Crystal-clear agent with full visibility into every decision point"""

    def __init__(self, model: str, max_iterations: int = 10):
        self.model = model
        self.max_iterations = max_iterations

    def execute_query(self, context: AgentContext, system_prompt: str) -> str:
        """Main execution loop with full instrumentation"""

        context.log_react_event("agent_execution_start", {
            "model": self.model,
            "max_iterations": self.max_iterations,
            "query": context.query
        })

        # Create ReAct executor
        react_executor = ReActExecutor(context, self.model)

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


def create_transparent_agent(model_name: str, max_iterations: int = 10) -> TransparentAgent:
    """Create a transparent agent with the specified model"""
    return TransparentAgent(model_name, max_iterations)