import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv
from jinja2 import Template
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from runner_utils import Workspace, RunConfig

# Console update control (toggled by --console-updates)
CONSOLE_UPDATES = False

# ----------------------------
# ReAct Models (Simplified)
# ----------------------------
# Note: Removing ThoughtStep and Observation models per colleague feedback
# Let PydanticAI handle intermediate steps via tool calls + messages
# Keep only FinalAnswer as the structured output type

# ----------------------------
# Output model (validated)
# ----------------------------
class FinalAnswer(BaseModel):
    answer: str = Field(description="Concise final answer addressing the guiding task.")
    artifacts: List[str] = Field(default_factory=list, description="Paths (relative to workspace) of useful outputs.")
    confidence: float = Field(ge=0.0, le=1.0, description="Self-rated confidence that the answer is sufficient.")

# ----------------------------
# Template System
# ----------------------------
def load_dataset_description(docs_path: str = "docs/2025_09_22_17_31_preliminary_data_analysis.md") -> str:
    """Load dataset description from markdown file."""
    docs_file = Path(docs_path)
    if docs_file.exists():
        return docs_file.read_text(encoding="utf-8")
    return "Dataset description not available."

def format_task_with_template(
    task: str,
    template_path: Optional[str] = None,
    dataset_path: str = "/Users/user/Projects/claude_data_agent/data/pipeline_data.parquet",
    dataset_description: Optional[str] = None,
    one_shot_examples: Optional[str] = None
) -> str:
    """Format task using Jinja2 template if provided, otherwise return task as-is."""

    # If no template specified, return original task
    if template_path is None:
        return task

    template_file = Path(template_path)
    if not template_file.exists():
        print(f"Warning: Template not found: {template_path}, using task as-is")
        return task

    # Load template
    template_content = template_file.read_text(encoding="utf-8")
    template = Template(template_content)

    # Load dataset description if not provided
    if dataset_description is None:
        dataset_description = load_dataset_description()

    # Format the task
    try:
        formatted_task = template.render(
            query=task,
            dataset_path=dataset_path,
            dataset_description=dataset_description,
            one_shot_examples=one_shot_examples,
            additional_libraries="",
            include_data_quality_notes=True,
            dataset_quality_issues="especially missing coordinates"
        )
        return formatted_task
    except Exception as e:
        print(f"Warning: Template formatting failed: {e}, using task as-is")
        return task

# ----------------------------
# Dependencies / Budget
# ----------------------------
@dataclass
class Deps:
    workspace: Workspace
    run_cfg: RunConfig
    tool_budget: int
    used_tools: int = 0

    def check_budget(self):
        if self.used_tools >= self.tool_budget:
            raise RuntimeError(f"Tool budget exceeded ({self.tool_budget}). Provide FinalAnswer now.")
        self.used_tools += 1

# ----------------------------
# Build the Agent
# ----------------------------
def build_agent(model: str) -> Agent[Deps, FinalAnswer]:
    system_instructions = (
        "You are an autonomous analyst using ReAct methodology: Think briefly, then call tools when needed, reflect on results.\n\n"

        "AVAILABLE TOOLS:\n"
        "- write_file: Create Python scripts or data files (always include console_message_update)\n"
        "- run_python: Execute Python scripts in workspace (always include console_message_update)\n"
        "- read_file: Read file contents (always include console_message_update)\n"
        "- list_files: See what files exist (always include console_message_update)\n\n"

        "PROCESS:\n"
        "1. Think briefly about what to do next\n"
        "2. Call a tool when needed (tools return results automatically).\n"
        "   When calling tools, ALWAYS set console_message_update to a short, user-friendly progress note (no jargon).\n"
        "3. Reflect on tool results and continue or conclude\n"
        "4. When done, output FinalAnswer with your conclusion\n\n"

        "GUIDELINES:\n"
        "- You have a budget of 5 tool calls maximum - prefer ≤3\n"
        "- Keep files small and use pure stdlib Python when possible\n"
        "- Keep messages concise\n"
        "- When task is complete, emit FinalAnswer (answer, artifacts, confidence)\n"
    )

    agent = Agent(
        model,
        deps_type=Deps,
        output_type=FinalAnswer,
        instructions=system_instructions,
    )

    # ---- Tools ----

    @agent.tool
    def write_file(ctx: RunContext[Deps], path: str, contents: str, executable: bool = False, console_message_update: Optional[str] = None) -> dict:
        """Write a text file inside the workspace. Keep it small. Params:
        - path: relative path, e.g. 'main.py'
        - contents: UTF-8 text
        - executable: set executable bit (rarely needed for .py)
        Returns minimal metadata and a small preview.
        """
        # Log tool call input
        log_react_event("tool_call_start", {
            "tool_name": "write_file",
            "parameters": {
                "path": path,
                "contents_preview": contents[:100] + "..." if len(contents) > 100 else contents,
                "contents_length": len(contents),
                "executable": executable,
                "console_message_update": console_message_update,
            },
            "tool_calls_used": ctx.deps.used_tools
        }, ctx.deps.workspace)

        ctx.deps.check_budget()
        result = ctx.deps.workspace.write_file(path, contents, executable=executable)

        # Log tool call output
        log_react_event("tool_call_complete", {
            "tool_name": "write_file",
            "result": result,
            "tool_calls_used": ctx.deps.used_tools
        }, ctx.deps.workspace)

        # Add concise summary for agent
        result["action"] = f"Created {path} ({result['bytes']} bytes)"
        return result

    @agent.tool
    def read_file(ctx: RunContext[Deps], path: str, console_message_update: Optional[str] = None) -> dict:
        """Read a file from the workspace with a truncated preview."""
        # Log tool call input
        log_react_event("tool_call_start", {
            "tool_name": "read_file",
            "parameters": {"path": path, "console_message_update": console_message_update},
            "tool_calls_used": ctx.deps.used_tools
        }, ctx.deps.workspace)

        ctx.deps.check_budget()
        result = ctx.deps.workspace.read_file(path)

        # Log tool call output
        log_react_event("tool_call_complete", {
            "tool_name": "read_file",
            "result": result,
            "tool_calls_used": ctx.deps.used_tools
        }, ctx.deps.workspace)

        result["action"] = f"Read {path} ({result['bytes']} bytes)"
        return result

    @agent.tool
    def list_files(ctx: RunContext[Deps], console_message_update: Optional[str] = None) -> list[dict]:
        """List files present in the workspace (path, bytes, sha256)."""
        # Log tool call input
        log_react_event("tool_call_start", {
            "tool_name": "list_files",
            "parameters": {"console_message_update": console_message_update},
            "tool_calls_used": ctx.deps.used_tools
        }, ctx.deps.workspace)

        ctx.deps.check_budget()
        files = ctx.deps.workspace.list_files()
        result = {"action": f"Listed {len(files)} files", "files": files}

        # Log tool call output
        log_react_event("tool_call_complete", {
            "tool_name": "list_files",
            "result": result,
            "tool_calls_used": ctx.deps.used_tools
        }, ctx.deps.workspace)

        return result

    @agent.tool
    def run_python(ctx: RunContext[Deps], entrypoint: str = "main.py", args: Optional[list[str]] = None, timeout_sec: Optional[int] = None, console_message_update: Optional[str] = None) -> dict:
        """Run a Python script within the workspace.
        - entrypoint: relative path to .py file (default 'main.py')
        - args: list of CLI args to pass
        - timeout_sec: override default timeout for this run
        Returns exit_code, wall_time, stdout/stderr previews, and log paths.
        """
        # Log tool call input
        log_react_event("tool_call_start", {
            "tool_name": "run_python",
            "parameters": {
                "entrypoint": entrypoint,
                "args": args or [],
                "timeout_sec": timeout_sec,
                "console_message_update": console_message_update,
            },
            "tool_calls_used": ctx.deps.used_tools
        }, ctx.deps.workspace)

        ctx.deps.check_budget()
        cfg = ctx.deps.run_cfg
        if timeout_sec is not None:
            cfg = RunConfig(timeout_sec=timeout_sec, allow_network=ctx.deps.run_cfg.allow_network, env=ctx.deps.run_cfg.env)
        result = ctx.deps.workspace.run_python(entrypoint=entrypoint, args=args or [], cfg=cfg)

        # Log tool call output
        log_react_event("tool_call_complete", {
            "tool_name": "run_python",
            "result": result,
            "tool_calls_used": ctx.deps.used_tools
        }, ctx.deps.workspace)

        result["action"] = f"Executed {entrypoint} (exit_code: {result.get('exit_code', 'unknown')})"
        return result

    return agent

# ----------------------------
# ReAct Controller with JSON Logging
# ----------------------------
def _format_tool_brief(event_type: str, data: dict) -> str:
    """Create a compact, human-friendly summary of a tool event."""
    tool = data.get("tool_name", "tool")
    params = data.get("parameters", {})
    result = data.get("result", {})

    if tool == "write_file":
        if event_type == "tool_call_complete" and result:
            return f"{event_type}: write_file path={result.get('path')} bytes={result.get('bytes')}"
        return f"{event_type}: write_file path={params.get('path')} bytes={params.get('contents_length')}"
    if tool == "read_file":
        if event_type == "tool_call_complete" and result:
            return f"{event_type}: read_file path={result.get('path')} bytes={result.get('bytes')}"
        return f"{event_type}: read_file path={params.get('path')}"
    if tool == "list_files":
        if event_type == "tool_call_complete" and result:
            count = len(result.get('files', [])) if isinstance(result, dict) else None
            return f"{event_type}: list_files count={count}"
        return f"{event_type}: list_files"
    if tool == "run_python":
        ep = params.get('entrypoint')
        args = params.get('args') or []
        if event_type == "tool_call_complete" and result:
            rc = result.get('exit_code')
            wall = result.get('wall_time_sec')
            return f"{event_type}: run_python {ep} exit={rc} time={wall}s"
        return f"{event_type}: run_python {ep} args={args}"
    return f"{event_type}: {tool}"


def log_react_event(event_type: str, data: dict, workspace: Workspace):
    """Log ReAct events to JSON file for observability"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "data": data
    }

    log_file = workspace.workspace / "react_log.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    # Optional console echo for progress visibility: show ONLY user-provided start messages.
    if CONSOLE_UPDATES and event_type == "tool_call_start":
        custom = data.get("parameters", {}).get("console_message_update")
        if isinstance(custom, str) and custom.strip():
            print(f"[progress] {custom.strip()}", flush=True)

class LoggingAgent:
    """Wrapper to add comprehensive logging to PydanticAI agent"""

    def __init__(self, agent: Agent, workspace: Workspace):
        self.agent = agent
        self.workspace = workspace
        self.message_count = 0
        self.conversation_history = []

    def run_sync(self, message: str, deps):
        """Run agent with comprehensive logging of full conversation"""
        self.message_count += 1

        # Log the user message/prompt
        user_message_data = {
            "message_id": self.message_count,
            "role": "user",
            "content": message,
            "tool_calls_used_before": deps.used_tools,
            "timestamp": datetime.now().isoformat()
        }

        log_react_event("llm_input", user_message_data, self.workspace)
        self.conversation_history.append(user_message_data)

        # Run the actual agent
        result = self.agent.run_sync(message, deps=deps)

        # Extract and display console update messages from the conversation
        self._extract_and_display_console_messages(result)

        # Log the complete LLM response
        llm_response_data = {
            "message_id": self.message_count,
            "role": "assistant",
            "output_type": type(result.output).__name__,
            "tool_calls_used_after": deps.used_tools,
            "tool_calls_in_this_interaction": deps.used_tools - user_message_data["tool_calls_used_before"],
            "timestamp": datetime.now().isoformat()
        }

        # Add output content based on type
        if hasattr(result.output, 'model_dump'):
            llm_response_data["output"] = result.output.model_dump()
            llm_response_data["output_serialized"] = result.output.model_dump_json()
        else:
            llm_response_data["output"] = str(result.output)

        # Try to capture conversation messages if available
        if hasattr(result, 'messages'):
            llm_response_data["full_conversation"] = [
                {
                    "role": msg.role,
                    "content": str(msg.content)[:500] + "..." if len(str(msg.content)) > 500 else str(msg.content),
                    "timestamp": getattr(msg, 'timestamp', None)
                }
                for msg in result.messages
            ]

        log_react_event("llm_output", llm_response_data, self.workspace)
        self.conversation_history.append(llm_response_data)

        # Log conversation summary
        log_react_event("conversation_state", {
            "message_id": self.message_count,
            "total_messages": len(self.conversation_history),
            "total_tool_calls": deps.used_tools,
            "conversation_summary": {
                "user_messages": len([m for m in self.conversation_history if m["role"] == "user"]),
                "assistant_messages": len([m for m in self.conversation_history if m["role"] == "assistant"]),
                "latest_output_type": type(result.output).__name__
            }
        }, self.workspace)

        return result

    def _extract_and_display_console_messages(self, result):
        """Extract and display console_update_message from LLM responses"""
        import re

        # Check if result has messages (conversation history)
        if not hasattr(result, 'messages'):
            return

        # Look for console_update_message patterns in assistant messages
        console_pattern = r'console_update_message:\s*["\']?([^"\'\n]+)["\']?'

        for msg in result.messages:
            if hasattr(msg, 'role') and msg.role == 'assistant':
                content = str(msg.content)
                matches = re.findall(console_pattern, content, re.IGNORECASE)

                for match in matches:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    formatted_message = f"[{timestamp}] {match.strip()}"

                    # Display with distinctive formatting
                    print(f"📊 {formatted_message}")

                    # Log the console update event
                    log_react_event("console_update", {
                        "timestamp": timestamp,
                        "message": match.strip(),
                        "extracted_from": "llm_response"
                    }, self.workspace)

def run_react_task(task: str, agent: Agent, deps: Deps, template_path: Optional[str] = None) -> FinalAnswer:
    """Run task using native PydanticAI flow with comprehensive JSON logging"""

    print("Starting Data Science Agent query")

    # Wrap agent with logging
    logging_agent = LoggingAgent(agent, deps.workspace)

    # Log task initiation
    log_react_event("task_start", {
        "task": task,
        "tool_budget": deps.tool_budget,
        "workspace": str(deps.workspace.workspace)
    }, deps.workspace)

    # Format task with template if provided
    formatted_task = format_task_with_template(task, template_path)

    # Simple task prompt - let agent think, call tools, and conclude naturally
    user_msg = formatted_task if template_path else f"TASK: {task}\n\nThink briefly about this task, then use available tools as needed. When complete, provide a FinalAnswer."

    try:
        # Let PydanticAI handle the entire ReAct flow naturally with logging
        result = logging_agent.run_sync(user_msg, deps)

        # Log successful completion
        log_react_event("task_complete", {
            "success": True,
            "tool_calls_used": deps.used_tools,
            "final_answer": result.output.model_dump() if hasattr(result.output, 'model_dump') else str(result.output)
        }, deps.workspace)

        # Intentionally keep console quiet here; progress is shown via tool updates.

        return result.output

    except RuntimeError as e:
        if "Tool budget exceeded" in str(e):
            print(f"\nTool budget exceeded after {deps.used_tools} calls")

            # Log budget exceeded
            log_react_event("task_failed", {
                "reason": "tool_budget_exceeded",
                "tool_calls_used": deps.used_tools,
                "tool_budget": deps.tool_budget
            }, deps.workspace)

            # Force final answer
            return FinalAnswer(
                answer=f"Task incomplete - exceeded tool budget of {deps.tool_budget} calls",
                artifacts=[],
                confidence=0.3
            )
        raise

# ----------------------------
# CLI
# ----------------------------
def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from yaml file."""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    return {}


def generate_run_metadata(task: str, workspace: Workspace, start_time: datetime, end_time: datetime,
                         model: str, template_path: Optional[str], max_tools: int, timeout: int,
                         tool_calls_used: int, success: bool, query_id: Optional[str] = None) -> dict:
    """Generate comprehensive metadata for a run."""
    import platform
    import sys
    import git
    from pathlib import Path

    # Get git information if available
    git_info = {}
    try:
        repo = git.Repo(search_parent_directories=True)
        git_info = {
            "commit_hash": repo.head.object.hexsha,
            "branch": repo.active_branch.name,
            "is_dirty": repo.is_dirty(),
            "remote_url": repo.remotes.origin.url if repo.remotes else None
        }
    except:
        git_info = {"error": "Git repository not found or accessible"}

    # Get dataset info if available
    dataset_info = {}
    data_path = workspace.workspace / "data"
    if data_path.exists() and data_path.is_symlink():
        try:
            real_path = data_path.readlink()
            if real_path.exists():
                stat = real_path.stat()
                dataset_info = {
                    "data_path": str(real_path),
                    "data_size_bytes": stat.st_size,
                    "data_modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
        except:
            dataset_info = {"error": "Could not access dataset information"}

    # Get workspace files created during run
    workspace_files = []
    if workspace.workspace.exists():
        for file_path in workspace.workspace.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                try:
                    stat = file_path.stat()
                    relative_path = file_path.relative_to(workspace.workspace)
                    workspace_files.append({
                        "path": str(relative_path),
                        "size_bytes": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                except:
                    pass  # Skip files we can't access

    metadata = {
        # Run identification
        "run_id": workspace.workspace.parent.name,
        "query_id": query_id,
        "timestamp": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds()
        },

        # Task information
        "task": {
            "text": task,
            "template_path": template_path,
            "template_exists": Path(template_path).exists() if template_path else False
        },

        # Execution parameters
        "execution": {
            "model": model,
            "max_tools": max_tools,
            "timeout_seconds": timeout,
            "tool_calls_used": tool_calls_used,
            "success": success
        },

        # Environment information
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "working_directory": str(Path.cwd())
        },

        # Version control
        "git": git_info,

        # Dataset information
        "dataset": dataset_info,

        # Workspace files generated
        "workspace_files": workspace_files,

        # Framework versions
        "versions": {
            "python": sys.version.split()[0],
            "platform_system": platform.system(),
            "platform_release": platform.release()
        }
    }

    return metadata


def save_run_metadata(metadata: dict, workspace: Workspace):
    """Save run metadata to metadata.json in the workspace (no console noise)."""
    metadata_path = workspace.workspace / "metadata.json"
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception:
        # Quietly ignore metadata save issues in console; logs contain details.
        pass


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Load configuration
    config = load_config()

    ap = argparse.ArgumentParser(description="Claude-Driven Local Runner (PydanticAI)")
    ap.add_argument("--task", type=str, help="Guiding task (one-liner).")
    ap.add_argument("--task-file", type=str, help="Path to a text file containing the guiding task.")
    ap.add_argument("--template", type=str, help="Path to Jinja2 template file for formatting task (optional).")
    ap.add_argument("--query-id", type=str, help="Query ID for batch processing (optional).")
    ap.add_argument("--config", type=str, default="config/config.yaml", help="Path to config YAML file.")

    # Override defaults with config values if available
    model_default = f"{config.get('model', {}).get('provider', 'anthropic')}:{config.get('model', {}).get('name', 'claude-sonnet-4-20250514')}"
    max_tools_default = config.get('agent', {}).get('max_tools', 5)
    timeout_default = config.get('agent', {}).get('timeout_sec', 60)
    workspace_default = config.get('workspace', {}).get('base_dir', '.runs')

    ap.add_argument("--model", type=str, default=model_default, help="Model string, e.g. 'anthropic:claude-sonnet-4-20250514'")
    ap.add_argument("--max-tools", type=int, default=max_tools_default, help="Maximum number of tool calls allowed.")
    ap.add_argument("--timeout", type=int, default=timeout_default, help="Seconds before a run_python call is killed.")
    ap.add_argument("--workspace", type=str, default=workspace_default, help="Base directory for runs; a unique subfolder is created per run.")
    ap.add_argument("--console-updates", action="store_true", help="Echo concise tool progress updates to console.")
    ap.add_argument("--no-console-updates", action="store_true", help="Disable tool progress updates.")
    args = ap.parse_args()

    # Apply console update preference
    global CONSOLE_UPDATES
    # Default ON; allow explicit disable
    CONSOLE_UPDATES = True
    if getattr(args, "no_console_updates", False):
        CONSOLE_UPDATES = False
    if getattr(args, "console_updates", False):
        CONSOLE_UPDATES = True

    # Load config again if a different config file was specified
    if args.config != "config/config.yaml":
        config = load_config(args.config)

    if not args.task and not args.task_file:
        ap.error("Provide --task or --task-file")

    task = args.task
    if args.task_file:
        task = Path(args.task_file).read_text(encoding="utf-8")

    # Prepare workspace/run
    base = Path(args.workspace)
    base.mkdir(parents=True, exist_ok=True)
    ws = Workspace(base_dir=base)

    # Dependencies
    allow_network = config.get('agent', {}).get('allow_network', False)
    deps = Deps(
        workspace=ws,
        run_cfg=RunConfig(timeout_sec=args.timeout, allow_network=allow_network, env={}),
        tool_budget=args.max_tools,
    )

    # Build agent
    agent = build_agent(args.model)

    # Track execution timing
    start_time = datetime.now()
    success = False

    try:
        # Run using ReAct methodology
        result = run_react_task(task.strip(), agent, deps, args.template)
        success = True
    finally:
        end_time = datetime.now()

        # Generate and save run metadata (inserted at end, after agent execution)
        metadata = generate_run_metadata(
            task=task.strip(),
            workspace=ws,
            start_time=start_time,
            end_time=end_time,
            model=args.model,
            template_path=args.template,
            max_tools=args.max_tools,
            timeout=args.timeout,
            tool_calls_used=deps.used_tools,
            success=success,
            query_id=args.query_id
        )
        save_run_metadata(metadata, ws)

    # Print only the answer string, not the full result dict
    print("\n=== FINAL ANSWER ===")
    answer_text = None
    # Prefer structured output
    if hasattr(result, "output"):
        try:
            if hasattr(result.output, "answer"):
                answer_text = result.output.answer
            elif isinstance(result.output, dict) and "answer" in result.output:
                answer_text = result.output["answer"]
        except Exception:
            answer_text = None

    # Fallback: read response.json in workspace if present
    if not answer_text:
        try:
            resp_path = ws.workspace / "response.json"
            if resp_path.exists():
                data = json.loads(resp_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "answer" in data:
                    answer_text = data["answer"]
        except Exception:
            answer_text = None

    # Final fallback: stringify output
    if not answer_text:
        try:
            answer_text = result.model_dump_json(indent=2)
        except Exception:
            answer_text = str(getattr(result, "output", ""))

    print(answer_text)
    print(f"\nRun directory: {ws.run_dir}\nWorkspace: {ws.workspace}\n")

if __name__ == "__main__":
    main()
