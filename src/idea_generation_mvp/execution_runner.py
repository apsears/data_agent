#!/usr/bin/env python3
"""
First step of execution refinement loop: Run portfolio seeds through Data Science Agent
"""

import json
import subprocess
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def load_portfolio() -> Dict[str, Any]:
    """Load the formalized portfolio seeds"""
    portfolio_file = Path("data/portfolio_seeds.json")
    if not portfolio_file.exists():
        raise FileNotFoundError(f"Portfolio file not found: {portfolio_file}")

    with open(portfolio_file, 'r') as f:
        return json.load(f)


def execute_seed_with_agent(seed: Dict[str, Any], query_id: str, refinement_mode: bool = False, previous_execution_summary: str = None, use_explicit_react: bool = True, enable_critic: bool = True) -> Dict[str, Any]:
    """Execute a single seed through the Data Science Agent"""

    query = seed["query"]
    print(f"\n🔄 Executing seed: {query}")
    print(f"   Method: {seed['method']}")
    print(f"   Assets: {', '.join(seed['assets'])}")
    if refinement_mode:
        print(f"   Mode: REFINEMENT")
    if use_explicit_react:
        print(f"   Agent: EXPLICIT REACT + {'CRITIC' if enable_critic else 'NO CRITIC'}")

    # Select template based on mode
    template = "templates/causal_inference_execution_refinement_prompt.txt" if refinement_mode else "templates/causal_analysis_agent_prompt.txt"

    # Build command for transparent agent executor
    cmd = [
        sys.executable,
        "transparent_agent_executor.py",
        "--task", "",  # Will be filled below
        "--query-id", query_id,
        "--template", template,
        "--max-tools", "10",
        "--timeout", "600",
        "--console-updates",
    ]

    # Add explicit ReAct flags if requested
    if use_explicit_react:
        cmd.append("--react-explicit")
        if enable_critic:
            cmd.append("--critic")

    # For refinement mode, we need to add context about previous execution
    if refinement_mode and previous_execution_summary:
        # Create a refined query that includes the context
        refined_query = f"EXECUTION REFINEMENT: {query}\n\nPREVIOUS EXECUTION SUMMARY: {previous_execution_summary}"
        cmd[cmd.index("--task") + 1] = refined_query
    else:
        cmd[cmd.index("--task") + 1] = query

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )

        execution_time = time.time() - start_time

        if result.returncode == 0:
            return {
                "success": True,
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "query_id": query_id
            }
        else:
            return {
                "success": False,
                "execution_time": execution_time,
                "error": f"Agent execution failed with exit code {result.returncode}",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "query_id": query_id
            }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "execution_time": 1800,
            "error": "Agent execution timed out after 30 minutes",
            "query_id": query_id
        }
    except Exception as e:
        return {
            "success": False,
            "execution_time": 0,
            "error": f"Failed to run agent: {str(e)}",
            "query_id": query_id
        }


def load_workspace_artifacts(query_id: str) -> Dict[str, Any]:
    """Load artifacts from the agent workspace"""

    # Find the workspace directory for this query
    runs_dir = Path(".runs")
    if not runs_dir.exists():
        return {"error": "No .runs directory found"}

    # Find most recent directory containing our query_id
    workspace_dir = None
    for run_dir in sorted(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if run_dir.is_dir():
            # Check if response.json exists and contains our query_id
            response_file = run_dir / "workspace" / "response.json"
            if response_file.exists():
                try:
                    with open(response_file, 'r') as f:
                        response_data = json.load(f)
                    if response_data.get("query_id") == query_id:
                        workspace_dir = run_dir
                        break
                except:
                    continue

    if not workspace_dir:
        return {"error": f"No workspace found for query_id: {query_id}"}

    artifacts = {
        "workspace_dir": str(workspace_dir),
        "files": {},
        "response": None,
        "metadata": None
    }

    # Load response.json
    response_file = workspace_dir / "workspace" / "response.json"
    if response_file.exists():
        try:
            with open(response_file, 'r') as f:
                artifacts["response"] = json.load(f)
        except Exception as e:
            artifacts["response_error"] = str(e)

    # Load metadata.json
    metadata_file = workspace_dir / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                artifacts["metadata"] = json.load(f)
        except Exception as e:
            artifacts["metadata_error"] = str(e)

    # Collect Python files and other artifacts
    workspace_files = workspace_dir / "workspace"
    if workspace_files.exists():
        for file_path in workspace_files.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.py', '.txt', '.md', '.json', '.csv']:
                rel_path = file_path.relative_to(workspace_files)
                try:
                    # Read small files completely, truncate large ones
                    content = file_path.read_text(encoding='utf-8')
                    if len(content) > 10000:  # Truncate very long files
                        content = content[:10000] + "\n... [truncated]"
                    artifacts["files"][str(rel_path)] = content
                except Exception as e:
                    artifacts["files"][str(rel_path)] = f"Error reading file: {e}"

    return artifacts


def generate_execution_report(seed: Dict[str, Any], execution_result: Dict[str, Any], artifacts: Dict[str, Any]) -> str:
    """Generate a markdown report of the execution"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Execution Report: {seed['query']}

**Generated:** {timestamp}
**Seed ID:** {seed['id']}
**Query ID:** {execution_result.get('query_id', 'unknown')}

## Original Seed Specification

**Query:** {seed['query']}
**Method:** {seed['method']}
**Assets:** {', '.join(seed['assets'])}
**Event Context:** {seed.get('event_context', 'None')}
**Technique Context:** {seed.get('technique_context', 'None')}
**Business Hook:** {seed.get('trader_hook', 'None')}
**Expected Artifacts:** {', '.join(seed.get('expected_artifacts', []))}
**Confidence:** {seed.get('confidence', 'Unknown')}

### Scores
- **Overall:** {seed['scores'].get('overall', 0):.2f}
- **Domain Relevance:** {seed['scores'].get('domain_relevance', 0):.2f}
- **Trader Value:** {seed['scores'].get('trader_value', 0):.2f}
- **Technical Rigor:** {seed['scores'].get('technical_rigor', 0):.2f}
- **Novelty:** {seed['scores'].get('novelty', 0):.2f}

## Execution Results

**Success:** {execution_result['success']}
**Execution Time:** {execution_result['execution_time']:.1f} seconds
"""

    if execution_result['success']:
        report += f"""
### Agent Response

{artifacts.get('response', {}).get('answer', 'No response captured')}

### Generated Code Artifacts
"""

        # Show Python files
        for filename, content in artifacts.get('files', {}).items():
            if filename.endswith('.py'):
                report += f"""
#### {filename}
```python
{content}
```
"""

        # Show other important files
        for filename, content in artifacts.get('files', {}).items():
            if not filename.endswith('.py') and filename != 'response.json':
                report += f"""
#### {filename}
```
{content}
```
"""

        # Show response metadata
        response = artifacts.get('response', {})
        if response:
            report += f"""
### Analysis Metadata

**Analysis Type:** {response.get('analysis_type', 'Unknown')}
**Methodology:** {response.get('methodology_explanation', 'Not provided')}
**Evidence Linkage:** {response.get('evidence_linkage', 'Not provided')}
**Limitations:** {response.get('limitations_uncertainties', 'Not provided')}
**Confidence:** {response.get('confidence', 'Not provided')}
"""

        # Show execution metadata
        metadata = artifacts.get('metadata', {})
        if metadata:
            report += f"""
### Execution Metadata

**Model Used:** {metadata.get('model', 'Unknown')}
**Template Used:** {metadata.get('template_used', 'Unknown')}
**Total Cost:** ${metadata.get('cost_info', {}).get('total_cost', 0):.4f}
**Input Tokens:** {metadata.get('cost_info', {}).get('input_tokens', 0):,}
**Output Tokens:** {metadata.get('cost_info', {}).get('output_tokens', 0):,}
"""

        report += f"""
### Workspace Directory

**Location:** {artifacts.get('workspace_dir', 'Unknown')}

**Files Created:**
"""
        for filename in artifacts.get('files', {}).keys():
            report += f"- {filename}\n"

    else:
        report += f"""
### Execution Error

**Error:** {execution_result.get('error', 'Unknown error')}

**STDOUT:**
```
{execution_result.get('stdout', 'No output')}
```

**STDERR:**
```
{execution_result.get('stderr', 'No errors')}
```
"""

    report += f"""
## Assessment for Refinement

### Data Discovery Assessment
- **Asset Identification:** [To be evaluated]
- **Timeline Precision:** [To be evaluated]
- **Data Coverage:** [To be evaluated]

### Methodological Implementation
- **Method Fidelity:** [To be evaluated]
- **Statistical Rigor:** [To be evaluated]
- **Artifact Quality:** [To be evaluated]

### Business Relevance
- **Decision Framework:** [To be evaluated]
- **Actionable Insights:** [To be evaluated]
- **Market Applicability:** [To be evaluated]

### Refinement Opportunities
[To be identified by LLM judge]

### Recommended Next Steps
[To be determined by refinement system]
"""

    return report


def create_previous_execution_summary(execution_report_path: str) -> str:
    """Extract key findings from previous execution report"""
    try:
        with open(execution_report_path, 'r') as f:
            report_content = f.read()

        # Extract key sections for summary
        summary_lines = []

        # Look for key results
        if "Primary Causal Impact Results" in report_content or "Key Analysis Findings" in report_content:
            lines = report_content.split('\n')
            in_results = False
            for line in lines:
                if "Primary Causal Impact Results" in line or "Key Analysis Findings" in line:
                    in_results = True
                    continue
                if in_results and line.startswith('###') and not line.startswith('### **'):
                    break
                if in_results and line.strip():
                    summary_lines.append(line.strip())

        # Create concise summary
        if summary_lines:
            summary = "\n".join(summary_lines[:10])  # First 10 relevant lines
            return f"Previous execution found: {summary}"
        else:
            return "Previous execution completed with causal analysis of Hurricane Beryl's impact on Gulf South Pipeline flows."

    except Exception as e:
        return f"Previous execution completed but summary unavailable: {e}"


def main():
    """Run the first seed through execution and generate report"""

    parser = argparse.ArgumentParser(description="Execute portfolio seeds through Data Science Agent")
    parser.add_argument("--refinement", action="store_true", help="Run in refinement mode using advanced prompt")
    parser.add_argument("--previous-report", type=str, help="Path to previous execution report for refinement context")
    parser.add_argument("--explicit-react", action="store_true", default=True, help="Use explicit ReAct loop with formal reasoning (default: True)")
    parser.add_argument("--no-explicit-react", action="store_true", help="Disable explicit ReAct loop")
    parser.add_argument("--critic", action="store_true", default=True, help="Enable automatic critic evaluation (default: True)")
    parser.add_argument("--no-critic", action="store_true", help="Disable critic evaluation")

    args = parser.parse_args()

    # Handle explicit react and critic flags
    use_explicit_react = args.explicit_react and not args.no_explicit_react
    enable_critic = args.critic and not args.no_critic and use_explicit_react

    mode_text = "REFINEMENT" if args.refinement else "INITIAL EXECUTION"
    agent_mode = "EXPLICIT REACT + CRITIC" if use_explicit_react and enable_critic else ("EXPLICIT REACT" if use_explicit_react else "STANDARD")
    print(f"🚀 Starting Execution Loop - {mode_text}")
    print(f"🤖 Agent Mode: {agent_mode}")
    print("=" * 50)

    # Load portfolio
    try:
        portfolio_data = load_portfolio()
        portfolio = portfolio_data["portfolio"]
        print(f"✅ Loaded portfolio with {len(portfolio)} seeds")
    except Exception as e:
        print(f"❌ Failed to load portfolio: {e}")
        return 1

    # Take the first (highest-scoring) seed
    first_seed = portfolio[0]
    mode_prefix = "refinement" if args.refinement else "execution"
    query_id = f"{mode_prefix}_{first_seed['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n🎯 Executing first seed:")
    print(f"   Query: {first_seed['query']}")
    print(f"   ID: {first_seed['id']}")
    print(f"   Query ID: {query_id}")

    # Prepare previous execution summary for refinement mode
    previous_execution_summary = None
    if args.refinement and args.previous_report:
        print(f"   Previous Report: {args.previous_report}")
        previous_execution_summary = create_previous_execution_summary(args.previous_report)
        print(f"   Summary: {previous_execution_summary[:100]}...")

    # Execute through Data Science Agent
    execution_result = execute_seed_with_agent(
        first_seed,
        query_id,
        refinement_mode=args.refinement,
        previous_execution_summary=previous_execution_summary,
        use_explicit_react=use_explicit_react,
        enable_critic=enable_critic
    )

    if execution_result['success']:
        print(f"✅ Execution completed successfully in {execution_result['execution_time']:.1f}s")

        # Load artifacts
        print("📂 Loading workspace artifacts...")
        artifacts = load_workspace_artifacts(query_id)

        if 'error' in artifacts:
            print(f"⚠️  Warning: {artifacts['error']}")
            artifacts = {"files": {}, "response": None, "metadata": None}
        else:
            print(f"✅ Loaded artifacts from {artifacts['workspace_dir']}")
            print(f"   Files: {len(artifacts['files'])}")

    else:
        print(f"❌ Execution failed: {execution_result['error']}")
        artifacts = {"files": {}, "response": None, "metadata": None}

    # Generate markdown report
    print("📝 Generating execution report...")
    report = generate_execution_report(first_seed, execution_result, artifacts)

    # Save report
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    report_type = "refinement_report" if args.refinement else "execution_report"
    report_file = Path(f"docs/{timestamp}_{report_type}_{first_seed['id']}.md")

    with open(report_file, 'w') as f:
        f.write(report)

    print(f"📄 Report saved to: {report_file}")

    print(f"\n{'=' * 50}")
    print("🎉 EXECUTION STEP COMPLETE!")
    print(f"{'=' * 50}")
    print(f"Next steps:")
    print(f"1. Review the execution report: {report_file}")
    print(f"2. Assess the quality and refinement potential")
    print(f"3. Build the judge for execution assessment")
    print(f"4. Implement LLM-based refinement")

    return 0


if __name__ == "__main__":
    sys.exit(main())