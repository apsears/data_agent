#!/usr/bin/env python3
"""
Cherry-pick script to generate EXAMPLES.md from specific run IDs.
Extracts task and answer from response.json files without editorializing.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def load_run_data(run_id: str) -> Dict[str, Any]:
    """Load response.json data for a given run ID."""
    run_path = Path(f".runs/{run_id}/workspace/response.json")

    if not run_path.exists():
        raise FileNotFoundError(f"No response.json found for run ID: {run_id}")

    with open(run_path, 'r') as f:
        return json.load(f)


def get_task_from_run(run_id: str) -> str:
    """Extract the original task/query from run artifacts."""
    # Try to find task in various locations
    run_dir = Path(f".runs/{run_id}")

    # Check for task in metadata if available
    metadata_path = run_dir / "workspace/metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            task_info = metadata.get('task', {})
            if isinstance(task_info, dict):
                return task_info.get('text', f"[Task for {run_id} not found]")
            return str(task_info)

    # Fallback: check run directory root for metadata
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            task_info = metadata.get('task', {})
            if isinstance(task_info, dict):
                return task_info.get('text', f"[Task for {run_id} not found]")
            return str(task_info)

    return f"[Task for {run_id} - check run logs]"


def generate_examples_md(run_ids: List[str], output_path: str = "EXAMPLES.md") -> None:
    """Generate EXAMPLES.md file from list of run IDs."""

    examples = []

    for run_id in run_ids:
        try:
            response_data = load_run_data(run_id)
            task = get_task_from_run(run_id)

            example = {
                'run_id': run_id,
                'task': task,
                'answer': response_data.get('answer', 'No answer found'),
                'confidence': response_data.get('confidence', 'N/A'),
                'numerical_results': response_data.get('numerical_results', {})
            }

            examples.append(example)

        except Exception as e:
            print(f"Error processing run {run_id}: {e}", file=sys.stderr)
            continue

    # Generate markdown content
    md_content = "# Examples\n\n"
    md_content += "Data science agent query results from actual runs.\n\n"

    for i, example in enumerate(examples, 1):
        md_content += f"## Example {i}\n\n"
        md_content += f"**Query:** {example['task']}\n\n"
        md_content += f"**Answer:** {example['answer']}\n\n"

        if example['numerical_results']:
            numerical = example['numerical_results']
            if 'primary_value' in numerical:
                md_content += f"**Primary Value:** {numerical['primary_value']}"
                if 'units' in numerical:
                    md_content += f" {numerical['units']}"
                md_content += "\n\n"


        md_content += f"**Run ID:** `{example['run_id']}`\n\n"
        md_content += "---\n\n"

    # Write to file
    with open(output_path, 'w') as f:
        f.write(md_content)

    print(f"Generated {output_path} with {len(examples)} examples")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python cherry-pick.py <run_id1> <run_id2> ... [--output EXAMPLES.md]")
        print("Example: python cherry-pick.py 20250926-063654-181dfcc2 20250926-063654-b3d6a2d6")
        sys.exit(1)

    # Parse arguments
    args = sys.argv[1:]
    output_path = "EXAMPLES.md"

    if "--output" in args:
        output_idx = args.index("--output")
        if output_idx + 1 < len(args):
            output_path = args[output_idx + 1]
            # Remove --output and its value from args
            args = args[:output_idx] + args[output_idx + 2:]
        else:
            print("Error: --output requires a filename", file=sys.stderr)
            sys.exit(1)

    run_ids = args

    if not run_ids:
        print("Error: No run IDs provided", file=sys.stderr)
        sys.exit(1)

    try:
        generate_examples_md(run_ids, output_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()