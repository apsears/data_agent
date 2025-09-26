#!/usr/bin/env python3
"""Simple single-query interface that reuses the batch runner workflow."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from run_batch_queries import run_single_query

DEFAULT_TEMPLATE = "templates/data_analysis_agent_prompt.txt"
DEFAULT_MODEL = "anthropic:claude-sonnet-4-20250514"
DEFAULT_MAX_TOOLS = 15
DEFAULT_TIMEOUT = 300
DEFAULT_CONFIG = "config/config.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a single data-science query using the batch workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_agent.py \"How many records are in the dataset?\"\n"
            "  python run_agent.py --model anthropic:claude-sonnet-4-20250514 --max-tools 20 \"Top pipeline companies by volume\"\n"
            "  python run_agent.py --no-console-updates --no-stream \"Summarize pipeline capacity in Texas\"\n"
        ),
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="Question to run through the agent. Use --task as an alternative flag.",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Alternative way to provide the question if quoting a positional argument is inconvenient.",
    )
    parser.add_argument(
        "--query-id",
        type=str,
        help="Optional query identifier; defaults to a timestamped value.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=DEFAULT_TEMPLATE,
        help=f"Template file to render the prompt (default: {DEFAULT_TEMPLATE}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use, e.g. 'anthropic:claude-sonnet-4-20250514' (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-tools",
        type=int,
        default=DEFAULT_MAX_TOOLS,
        help=f"Maximum number of tool calls allowed (default: {DEFAULT_MAX_TOOLS}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout in seconds for tool executions (default: {DEFAULT_TIMEOUT}).",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        help="Base directory for runs (passed through to the underlying agent).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Configuration file for the underlying agent (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--no-console-updates",
        action="store_true",
        help="Disable progress updates emitted by the agent tools.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable live streaming of the agent output (useful for clean logs).",
    )

    return parser


def resolve_query(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if args.task and args.query:
        parser.error("Provide either a positional query or --task, not both.")

    task = args.task or args.query
    if not task:
        parser.error("A query is required. Supply it positionally or with --task.")

    return task.strip()


def build_base_args(args: argparse.Namespace) -> list[str]:
    base_args = [
        "--template",
        args.template,
        "--max-tools",
        str(args.max_tools),
        "--timeout",
        str(args.timeout),
        "--model",
        args.model,
    ]

    if not args.no_console_updates:
        base_args.append("--console-updates")

    if args.workspace:
        base_args.extend(["--workspace", args.workspace])

    if args.config and args.config != DEFAULT_CONFIG:
        base_args.extend(["--config", args.config])

    return base_args


def load_response_data(run_directory: Optional[str]) -> Optional[Dict[str, Any]]:
    if not run_directory:
        return None

    response_path = Path(run_directory) / "workspace" / "response.json"
    if not response_path.exists():
        return None

    try:
        return json.loads(response_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def display_final_result(result: Dict[str, Any]) -> None:
    # The agent already displays the final answer and run directory info
    # Only show errors if the run failed
    if not result.get("success", False):
        error = result.get("error") or "The agent run failed."
        print(f"\n❌ {error}")
        if result.get("execution_time") is not None:
            print(f"Execution time: {result['execution_time']:.1f}s")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    task = resolve_query(args, parser)
    query_id = args.query_id or f"single-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    base_args = build_base_args(args)

    query = {"id": query_id, "category": "cli", "query": task}

    try:
        result = run_single_query(query, base_args, stream=not args.no_stream, worker_id=0)
    except KeyboardInterrupt:
        print("\nQuery interrupted by user")
        sys.exit(130)
    except Exception as exc:
        print(f"\nUnexpected error: {exc}")
        sys.exit(1)

    display_final_result(result)

    sys.exit(0 if result.get("success", False) else 1)


if __name__ == "__main__":
    main()
