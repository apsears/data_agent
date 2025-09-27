#!/usr/bin/env python3
"""
Agent Tools Library - Tool functions and logging for PydanticAI agent
"""

import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from pydantic_ai import RunContext


class ReActStepLogger:
    """Enhanced ReAct conversation logger"""

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


def write_file_and_run_python(ctx: RunContext, file_path: str, content: str) -> str:
    """Write a Python script file and immediately execute it. This fused operation reduces tool calls."""
    state = ctx.deps
    start_time = time.time()

    # Generate unique task ID for this execution
    task_id = f"{int(time.time() * 1000000)}-{file_path.replace('.py', '').replace('/', '_')}"

    # Log tool call start
    state.log_react_event("tool_call_start", {
        "tool_name": "write_file_and_run_python",
        "file_path": file_path,
        "content_length": len(content),
        "task_id": task_id
    })

    try:
        # Check if file already exists and prevent overwriting
        full_path = state.workspace_dir / file_path
        if full_path.exists():
            error_msg = f"❌ File {file_path} already exists. Use a different filename to avoid overwriting existing scripts."
            state.log_react_event("tool_call_complete", {
                "tool_name": "write_file_and_run_python",
                "success": False,
                "error": error_msg,
                "duration": time.time() - start_time
            })
            return error_msg

        # Write file to workspace
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Create logs directory in workspace parent
        logs_dir = state.workspace_dir.parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Setup log file paths
        stdout_log = logs_dir / f"stdout-{task_id}.log"
        stderr_log = logs_dir / f"stderr-{task_id}.log"

        # Run python script using subprocess
        try:
            result = subprocess.run(
                [sys.executable, file_path],
                cwd=str(state.workspace_dir),
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
                if result.stderr:
                    error_msg += f"\nSTDERR: {result.stderr}"
                return f"❌ {error_msg}\n\n{output}"

            return output

        except subprocess.TimeoutExpired:
            error_msg = f"Script execution timed out after 300 seconds"
            state.log_react_event("tool_call_complete", {
                "tool_name": "write_file_and_run_python",
                "success": False,
                "error": error_msg,
                "duration": time.time() - start_time
            })
            return f"❌ {error_msg}"

        except Exception as e:
            error_msg = f"Script execution failed: {str(e)}"
            state.log_react_event("tool_call_complete", {
                "tool_name": "write_file_and_run_python",
                "success": False,
                "error": error_msg,
                "duration": time.time() - start_time
            })
            return f"❌ {error_msg}"

    except Exception as e:
        error_msg = f"Failed to write file {file_path}: {str(e)}"
        state.log_react_event("tool_call_complete", {
            "tool_name": "write_file_and_run_python",
            "success": False,
            "error": error_msg,
            "duration": time.time() - start_time
        })
        return f"❌ {error_msg}"


def read_file(ctx: RunContext, file_path: str) -> str:
    """Read contents of a file in the workspace."""
    state = ctx.deps

    state.log_react_event("tool_call_start", {
        "tool_name": "read_file",
        "file_path": file_path
    })

    try:
        full_path = state.workspace_dir / file_path
        if not full_path.exists():
            result = f"❌ File not found: {file_path}"
        else:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            result = f"📖 Content of {file_path}:\n\n{content}"

        state.log_react_event("tool_call_complete", {
            "tool_name": "read_file",
            "success": full_path.exists() if 'full_path' in locals() else False,
            "file_path": file_path,
            "content_length": len(content) if 'content' in locals() else 0
        })

        return result

    except Exception as e:
        error_msg = f"Failed to read file {file_path}: {str(e)}"
        state.log_react_event("tool_call_complete", {
            "tool_name": "read_file",
            "success": False,
            "error": error_msg
        })
        return f"❌ {error_msg}"


def list_files(ctx: RunContext) -> str:
    """List all files in the workspace directory."""
    state = ctx.deps

    state.log_react_event("tool_call_start", {
        "tool_name": "list_files"
    })

    try:
        files = []
        for item in state.workspace_dir.iterdir():
            if item.is_file():
                files.append(f"📄 {item.name}")
            elif item.is_dir():
                files.append(f"📁 {item.name}/")

        if not files:
            result = "📂 Workspace is empty"
        else:
            result = f"📂 Files in workspace:\n" + "\n".join(sorted(files))

        state.log_react_event("tool_call_complete", {
            "tool_name": "list_files",
            "success": True,
            "file_count": len([f for f in files if f.startswith("📄")])
        })

        return result

    except Exception as e:
        error_msg = f"Failed to list files: {str(e)}"
        state.log_react_event("tool_call_complete", {
            "tool_name": "list_files",
            "success": False,
            "error": error_msg
        })
        return f"❌ {error_msg}"


def save_conversation_log(workspace_dir: Path, query_id: str, query: str, all_messages, logger: ReActStepLogger):
    """Save complete conversation log to separate file"""
    conversation_log = {
        "query_id": query_id,
        "query": query,
        "total_messages": len(all_messages),
        "complete_conversation": [str(msg) for msg in all_messages],
        "detailed_breakdown": logger._extract_message_breakdown(all_messages)
    }

    conversation_file = workspace_dir / "complete_conversation_log.json"
    with open(conversation_file, 'w', encoding='utf-8') as f:
        json.dump(conversation_log, f, indent=2, default=str)