#!/usr/bin/env python3
"""
Batch Query Runner for Pipeline Data Analysis

Runs multiple queries from a JSON file using the ReAct framework.
"""

import argparse
import json
import subprocess
import sys
import time
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
from jinja2 import Template
import tiktoken
import pandas as pd
from dotenv import load_dotenv
import anthropic
import openai


def load_queries(queries_file: str) -> Dict[str, Any]:
    """Load queries from JSON file."""
    with open(queries_file, 'r') as f:
        return json.load(f)


def load_pricing_data(model: str) -> Dict[str, Dict[str, float]]:
    """Load pricing data from appropriate TSV file based on model provider."""
    if model.startswith("anthropic:"):
        pricing_file = "config/anthropic_pricing.tsv"
    else:
        pricing_file = "config/openai_pricing.tsv"

    pricing_path = Path(pricing_file)

    if not pricing_path.exists():
        raise FileNotFoundError(f"Pricing file not found: {pricing_file}")

    df = pd.read_csv(pricing_path, sep='\t')
    pricing = {}

    for _, row in df.iterrows():
        model_name = row['Model'].lower()
        try:
            input_str = str(row['Input']).replace('$', '')
            output_str = str(row['Output']).replace('$', '')

            # Skip rows with missing data (represented as '-')
            if input_str == '-' or output_str == '-':
                continue

            input_price = float(input_str)
            output_price = float(output_str)
            pricing[model_name] = {
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
            "gpt-3.5-turbo": "cl100k_base"
        }

        encoding_name = encoding_map.get(model.replace("anthropic:", "").replace("openai:", ""), "o200k_base")
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to rough estimation
        return len(text) // 4


def select_template_for_query(query: Dict[str, Any], default_template: str) -> str:
    """Select appropriate template based on query analysis type."""
    analysis_type = query.get("analysis_type", "data_analysis")

    template_map = {
        "factual": "templates/factual_analysis_agent_prompt.txt",
        "pattern": "templates/pattern_analysis_agent_prompt.txt",
        "causal": "templates/causal_analysis_agent_prompt.txt"
    }

    selected_template = template_map.get(analysis_type, default_template)

    # Verify template exists, fallback to default if not
    if not Path(selected_template).exists():
        print(f"[WARNING] Template {selected_template} not found, using default: {default_template}")
        return default_template

    return selected_template


def run_single_query(query: Dict[str, Any], base_args: List[str], stream: bool = True, worker_id: int = 0) -> Dict[str, Any]:
    """Run a single query using run_agent.py and capture results.

    When stream=True, child stdout is printed live to the console while being captured
    for later parsing. This gives real-time progress visibility.
    """
    worker_prefix = f"[Worker-{worker_id}]" if worker_id > 0 else ""
    print(f"\n{'='*60}")
    print(f"{worker_prefix}Running Query {query['id']}: {query['category']}")
    print(f"{worker_prefix}Query: {query['query']}")
    print(f"{'='*60}")

    # Select appropriate template based on analysis type
    selected_template = None
    modified_args = []
    for i, arg in enumerate(base_args):
        if arg == "--template" and i + 1 < len(base_args):
            default_template = base_args[i + 1]
            selected_template = select_template_for_query(query, default_template)
            modified_args.extend(["--template", selected_template])
            # Skip the next argument since we handled it
            continue
        elif i > 0 and base_args[i - 1] == "--template":
            # Skip this argument since it was handled above
            continue
        else:
            modified_args.append(arg)

    if selected_template:
        print(f"{worker_prefix}Using template: {selected_template}")

    # Build command
    cmd = [
        sys.executable, "-u", "transparent_agent_executor.py",  # -u ensures unbuffered child output for live streaming
        "--task", query["query"],
        "--query-id", query["id"],
        *modified_args
    ]

    start_time = time.time()

    try:
        # Launch the command and optionally stream stdout in real time
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )

        captured_lines: List[str] = []
        timeout_sec = 600  # 10 minute timeout per query

        # Read incrementally to allow live progress
        while True:
            line = proc.stdout.readline() if proc.stdout else ''
            if line:
                captured_lines.append(line)
                if stream:
                    prefixed_line = f"{worker_prefix}{line}" if worker_id > 0 and not line.startswith('[') else line
                    print(prefixed_line, end='', flush=True)
            elif proc.poll() is not None:
                # Process finished; drain any remaining output
                if proc.stdout:
                    remainder = proc.stdout.read()
                    if remainder:
                        captured_lines.append(remainder)
                        if stream:
                            prefixed_remainder = f"{worker_prefix}{remainder}" if worker_id > 0 and not remainder.startswith('[') else remainder
                            print(prefixed_remainder, end='', flush=True)
                break

            # Enforce timeout
            if time.time() - start_time > timeout_sec:
                proc.kill()
                raise subprocess.TimeoutExpired(cmd, timeout_sec)

        execution_time = time.time() - start_time

        full_stdout = ''.join(captured_lines)
        stdout_lines = full_stdout.strip().split('\n') if full_stdout else []

        # Parse the output to extract final answer
        final_answer_start = None
        for i, line in enumerate(stdout_lines):
            if "=== FINAL ANSWER ===" in line:
                final_answer_start = i + 1
                break

        final_answer = ""
        if final_answer_start is not None:
            final_answer = '\n'.join(stdout_lines[final_answer_start:])

        # Extract run directory for artifacts
        run_directory = None
        for line in stdout_lines:
            if "Run directory:" in line:
                run_directory = line.split("Run directory:")[-1].strip()
                break

        # Load detailed timing information if available
        detailed_timing = None
        cost_info = None
        if run_directory:
            timing_file = Path(run_directory) / "timing_breakdown.json"
            if timing_file.exists():
                try:
                    with open(timing_file, 'r') as f:
                        detailed_timing = json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass  # Timing file exists but couldn't be loaded

            # Load cost information from metadata.json if available
            metadata_file = Path(run_directory) / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        cost_info = metadata.get("cost_info")
                except (json.JSONDecodeError, IOError):
                    pass  # Metadata file exists but couldn't be loaded

        result = {
            "query_id": query["id"],
            "query": query["query"],
            "category": query["category"],
            "success": (proc.returncode or 0) == 0,
            "execution_time": execution_time,
            "detailed_timing": detailed_timing,
            "final_answer": final_answer,
            "run_directory": run_directory,
            "stdout": full_stdout,
            "stderr": "",  # merged into stdout
            "return_code": proc.returncode or 0,
        }

        # Add cost information if available
        if cost_info:
            result["cost_info"] = cost_info

        return result

    except subprocess.TimeoutExpired:
        return {
            "query_id": query["id"],
            "query": query["query"],
            "category": query["category"],
            "success": False,
            "execution_time": time.time() - start_time,
            "error": "Query execution timed out (10 minutes)",
            "final_answer": None,
            "run_directory": None,
            "return_code": -1
        }
    except Exception as e:
        return {
            "query_id": query["id"],
            "query": query["query"],
            "category": query["category"],
            "success": False,
            "execution_time": time.time() - start_time,
            "error": str(e),
            "final_answer": None,
            "run_directory": None,
            "return_code": -1
        }


def _calculate_efficiency_score(execution_time: float) -> int:
    """Calculate efficiency score based on execution time.

    Scoring scale:
    - <60s: 5
    - 60-120s: 4
    - 120-180s: 3
    - 180-240s: 2
    - 240-300s: 1
    - >300s: 0
    """
    if execution_time < 60:
        return 5
    elif execution_time < 120:
        return 4
    elif execution_time < 180:
        return 3
    elif execution_time < 240:
        return 2
    elif execution_time < 300:
        return 1
    else:
        return 0


def _load_rubric(analysis_type: str) -> Dict[str, Any]:
    """Load evaluation rubric for the specified analysis type."""
    rubric_path = Path(f"config/rubrics/{analysis_type}_analysis_rubric.yaml")
    if not rubric_path.exists():
        # Fallback to factual rubric if specific one doesn't exist
        rubric_path = Path("config/rubrics/factual_analysis_rubric.yaml")
        if not rubric_path.exists():
            raise FileNotFoundError(f"No rubric found for {analysis_type} analysis")

    with open(rubric_path, 'r') as f:
        return yaml.safe_load(f)


def _load_judging_template(config: Dict[str, Any]) -> Template:
    """Load and parse the judging template."""
    # Use rubric-based template if available, fall back to original
    rubric_template_path = Path("templates/rubric_based_judge_prompt.txt")
    if rubric_template_path.exists():
        template_path = rubric_template_path
    else:
        template_path = Path(config["judging"]["template"])

    if not template_path.exists():
        raise FileNotFoundError(f"Judging template not found: {template_path}")
    return Template(template_path.read_text(encoding="utf-8"))


def _extract_actual_answer(result: Dict[str, Any]) -> str:
    """Extract the actual answer from query result, preferring response.json."""
    run_dir = result.get("run_directory")

    if run_dir:
        response_file = Path(run_dir) / "workspace" / "response.json"
        if response_file.exists():
            try:
                response_data = json.loads(response_file.read_text(encoding="utf-8"))
                return response_data.get("answer", "")
            except json.JSONDecodeError:
                pass  # Fall through to fallback

    # Fallback to final_answer field
    return result.get("final_answer", "")


def _call_judging_llm(judging_prompt: str, judging_model: str) -> tuple[str, int]:
    """Make API call to judging LLM and return response text and output tokens."""
    load_dotenv()

    # Define the JSON schema for structured output
    json_schema = {
        "name": "judge_result",
        "description": "Structured judging result with accuracy score and detailed evaluation",
        "schema": {
            "type": "object",
            "properties": {
                "accuracy_score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 5,
                    "description": "Accuracy score from 0-5"
                },
                "explanation": {
                    "type": "string",
                    "description": "Detailed explanation of scoring rationale"
                },
                "numerical_comparison": {
                    "type": "string",
                    "description": "Specific comparison of key numbers/metrics"
                },
                "methodology_assessment": {
                    "type": "string",
                    "description": "Evaluation of analytical approach"
                },
                "completeness_check": {
                    "type": "string",
                    "description": "Assessment of whether query was fully addressed"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in this evaluation (0.0-1.0)"
                }
            },
            "required": ["accuracy_score", "explanation", "numerical_comparison", "methodology_assessment", "completeness_check", "confidence"]
        }
    }

    if judging_model.startswith("anthropic:"):
        try:
            client = anthropic.Anthropic()
            model_name = judging_model.replace("anthropic:", "")
            response = client.messages.create(
                model=model_name,
                max_tokens=2000,
                messages=[{"role": "user", "content": judging_prompt}],
                tools=[{
                    "name": "judge_result",
                    "description": "Structured judging result with accuracy score and detailed evaluation",
                    "input_schema": json_schema["schema"]
                }],
                tool_choice={"type": "tool", "name": "judge_result"}
            )
            # Extract structured output from tool use
            tool_use = response.content[0]
            if hasattr(tool_use, 'input'):
                # tool_use.input is already a dictionary, no need to serialize
                return tool_use.input, response.usage.output_tokens
            else:
                return response.content[0].text, response.usage.output_tokens
        except Exception as e:
            raise

    elif judging_model.startswith("openai:"):
        client = openai.OpenAI()
        model_name = judging_model.replace("openai:", "")
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=2000,
            messages=[{"role": "user", "content": judging_prompt}],
            functions=[json_schema],
            function_call={"name": "judge_result"}
        )
        if response.choices[0].message.function_call:
            return response.choices[0].message.function_call.arguments, response.usage.completion_tokens
        else:
            return response.choices[0].message.content, response.usage.completion_tokens

    else:
        raise ValueError(f"Unsupported judging model provider: {judging_model}")


def _parse_judging_response(judge_response) -> Dict[str, Any]:
    """Parse JSON from structured judging response."""
    # Handle both dictionary (API) and string (legacy) responses
    if isinstance(judge_response, dict):
        return judge_response
    elif isinstance(judge_response, str):
        try:
            return json.loads(judge_response)
        except json.JSONDecodeError:
            # Fallback for any edge cases
            return {}
    else:
        return {}


def _calculate_judging_cost(input_tokens: int, output_tokens: int, judging_model: str) -> Dict[str, Any]:
    """Calculate the cost of judging based on token usage and model pricing."""
    judging_model_name = judging_model.replace("anthropic:", "").replace("openai:", "")
    pricing_data = load_pricing_data(judging_model)

    if judging_model_name not in pricing_data:
        raise ValueError(f"Pricing not found for model: {judging_model_name}")

    model_pricing = pricing_data[judging_model_name]
    input_cost = input_tokens * model_pricing["input_per_1m"] / 1_000_000
    output_cost = output_tokens * model_pricing["output_per_1m"] / 1_000_000

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_cost": input_cost + output_cost
    }


def judge_single_result(result: Dict[str, Any], query_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Judge a single query result using LLM evaluation with rubric-based assessment."""

    # Early exit if query failed
    if not result["success"]:
        return {
            "judging_performed": False,
            "accuracy_score": 0,
            "explanation": "Query execution failed, cannot evaluate",
            "judging_cost": {"input_tokens": 0, "output_tokens": 0, "total_cost": 0.0}
        }

    try:
        # Load template and extract answer
        template = _load_judging_template(config)
        actual_answer = _extract_actual_answer(result)

        # Determine analysis type and load appropriate rubric
        analysis_type = query_data.get("analysis_type", "factual")
        rubric = _load_rubric(analysis_type)

        # Prepare rubric content for template
        rubric_content = f"""
# {rubric['name']}

**Description**: {rubric['description']}

## Scoring Scale:
"""
        for scale_item in rubric['scoring_scale']:
            rubric_content += f"- **{scale_item['score']}** ({scale_item['label']}): {scale_item['description']}\n"

        rubric_content += "\n## Evaluation Criteria:\n"
        for criterion_name, criterion_data in rubric['evaluation_criteria'].items():
            rubric_content += f"\n### {criterion_name.replace('_', ' ').title()} (Weight: {criterion_data['weight']})\n"
            rubric_content += f"{criterion_data['description']}\n\n"
            rubric_content += "**Scoring Guidelines:**\n"
            for score, guideline in criterion_data['scoring_guidelines'].items():
                rubric_content += f"- **{score}**: {guideline}\n"

        rubric_content += f"\n## Red Flags:\n"
        for flag in rubric['red_flags']:
            rubric_content += f"- {flag}\n"

        rubric_content += f"\n## Strengths to Recognize:\n"
        for strength in rubric['strengths_to_recognize']:
            rubric_content += f"- {strength}\n"

        # Calculate efficiency score based on execution time
        execution_time = result.get("execution_time", 0)
        efficiency_score = _calculate_efficiency_score(execution_time)

        # Render judging prompt with rubric
        judging_prompt = template.render(
            query=query_data["query"],
            expected_answer=query_data.get("expected_answer", "No reference answer provided"),
            actual_answer=actual_answer,
            analysis_type=analysis_type,
            rubric_content=rubric_content,
            rubric_criteria=rubric['evaluation_criteria'],
            execution_time=execution_time,
            efficiency_score=efficiency_score
        )

        # Calculate input tokens and make API call
        judging_model = config["judging"]["model"]
        judging_model_name = judging_model.replace("anthropic:", "").replace("openai:", "")
        input_tokens = count_tokens(judging_prompt, judging_model_name)

        start_time = time.time()
        judge_response, output_tokens = _call_judging_llm(judging_prompt, judging_model)
        judging_time = time.time() - start_time


        # Parse response and calculate costs
        judging_data = _parse_judging_response(judge_response)
        judging_cost = _calculate_judging_cost(input_tokens, output_tokens, judging_model)

        return {
            "judging_performed": True,
            "judging_time": judging_time,
            "accuracy_score": judging_data.get("accuracy_score", -1),
            "efficiency_score": efficiency_score,
            "execution_time": execution_time,
            "explanation": judging_data.get("explanation", "No explanation provided"),
            "numerical_comparison": judging_data.get("numerical_comparison", ""),
            "methodology_assessment": judging_data.get("methodology_assessment", ""),
            "completeness_check": judging_data.get("completeness_check", ""),
            "confidence": judging_data.get("confidence", 0.0),
            "judging_cost": judging_cost,
            "raw_judging_output": str(judge_response)  # Ensure this is always a string
        }

    except Exception as e:
        return {
            "judging_performed": False,
            "error": f"Judging failed: {str(e)}",
            "judging_cost": {"input_tokens": 0, "output_tokens": 0, "total_cost": 0.0}
        }


def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save batch results to JSON file."""
    successful_results = [r for r in results if r["success"]]

    # Calculate timing aggregates
    timing_summary = None
    if successful_results:
        total_tool_time = 0
        total_thinking_time = 0
        tool_breakdown = {}

        results_with_timing = [r for r in successful_results if r.get("detailed_timing")]

        for result in results_with_timing:
            dt = result["detailed_timing"]
            total_tool_time += dt.get("total_tool_time", 0)
            total_thinking_time += dt.get("thinking_time", 0)

            for tool_name, tool_stats in dt.get("tool_breakdown", {}).items():
                if tool_name not in tool_breakdown:
                    tool_breakdown[tool_name] = {"count": 0, "total_duration": 0, "successes": 0, "failures": 0}

                tool_breakdown[tool_name]["count"] += tool_stats["count"]
                tool_breakdown[tool_name]["total_duration"] += tool_stats["total_duration"]
                tool_breakdown[tool_name]["successes"] += tool_stats["successful_calls"]
                tool_breakdown[tool_name]["failures"] += tool_stats["failed_calls"]

        if results_with_timing:
            timing_summary = {
                "queries_with_detailed_timing": len(results_with_timing),
                "total_tool_time": total_tool_time,
                "total_thinking_time": total_thinking_time,
                "avg_tool_time_per_query": total_tool_time / len(results_with_timing),
                "avg_thinking_time_per_query": total_thinking_time / len(results_with_timing),
                "tool_breakdown": tool_breakdown
            }

    batch_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": len(results),
        "successful_queries": len(successful_results),
        "failed_queries": len([r for r in results if not r["success"]]),
        "total_execution_time": sum(r["execution_time"] for r in results),
        "timing_summary": timing_summary,
        "results": results
    }

    # Add agent cost summary if any results have cost info
    results_with_cost = [r for r in results if r.get("cost_info")]
    if results_with_cost:
        total_agent_cost = sum(r["cost_info"]["total_cost"] for r in results_with_cost)
        total_input_tokens = sum(r["cost_info"]["input_tokens"] for r in results_with_cost)
        total_output_tokens = sum(r["cost_info"]["output_tokens"] for r in results_with_cost)
        batch_summary["agent_cost_summary"] = {
            "total_queries_with_cost": len(results_with_cost),
            "total_agent_cost": total_agent_cost,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "avg_cost_per_query": total_agent_cost / len(results_with_cost)
        }

    # Add judging summary if any results were judged
    judged_results = [r for r in results if r.get("judging", {}).get("judging_performed", False)]
    if judged_results:
        total_judging_cost = sum(r["judging"]["judging_cost"]["total_cost"] for r in judged_results)
        avg_score = sum(r["judging"]["accuracy_score"] for r in judged_results) / len(judged_results)

        batch_summary["judging_summary"] = {
            "total_judged": len(judged_results),
            "average_accuracy_score": avg_score,
            "total_judging_cost": total_judging_cost,
            "total_input_tokens": sum(r["judging"]["judging_cost"]["input_tokens"] for r in judged_results),
            "total_output_tokens": sum(r["judging"]["judging_cost"]["output_tokens"] for r in judged_results)
        }

    # Add combined cost summary if both agent and judging costs are available
    if results_with_cost and judged_results:
        total_combined_cost = (batch_summary["agent_cost_summary"]["total_agent_cost"] +
                             batch_summary["judging_summary"]["total_judging_cost"])
        batch_summary["total_cost_summary"] = {
            "total_agent_cost": batch_summary["agent_cost_summary"]["total_agent_cost"],
            "total_judging_cost": batch_summary["judging_summary"]["total_judging_cost"],
            "total_combined_cost": total_combined_cost
        }

    with open(output_file, 'w') as f:
        json.dump(batch_summary, f, indent=2)


def run_query_with_judging(query_info: tuple) -> Dict[str, Any]:
    """Helper function to run a single query with judging in parallel."""
    query, base_args, config, judging_enabled, no_stream, worker_id = query_info

    # Run the query
    result = run_single_query(query, base_args, stream=not no_stream, worker_id=worker_id)

    # Add judging if enabled and successful
    if judging_enabled and result["success"]:
        worker_prefix = f"[Worker-{worker_id}]" if worker_id > 0 else ""
        print(f"{worker_prefix}  Judging result...")
        judging_result = judge_single_result(result, query, config)
        result["judging"] = judging_result

    return result


def run_queries_parallel(queries: List[Dict[str, Any]], base_args: List[str],
                        config: Dict[str, Any], judging_enabled: bool,
                        no_stream: bool, workers: int) -> List[Dict[str, Any]]:
    """Run queries in parallel using ThreadPoolExecutor."""

    if workers == 1:
        # Serial execution (original behavior)
        results = []
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] Processing query {query['id']}...")
            result = run_query_with_judging((query, base_args, config, judging_enabled, no_stream, 0))
            results.append(result)

            # Brief pause between queries in serial mode
            if i < len(queries):
                time.sleep(2)

        return results

    # Parallel execution
    print(f"\nRunning {len(queries)} queries in parallel with {workers} workers...")

    # Prepare query info tuples with worker IDs
    query_infos = []
    for i, query in enumerate(queries):
        worker_id = (i % workers) + 1
        query_infos.append((query, base_args, config, judging_enabled, no_stream, worker_id))

    results = [None] * len(queries)  # Preserve order

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all queries
        future_to_index = {
            executor.submit(run_query_with_judging, query_info): i
            for i, query_info in enumerate(query_infos)
        }

        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
                print(f"\n✅ Query {queries[index]['id']} completed")
            except Exception as e:
                print(f"\n❌ Query {queries[index]['id']} failed: {e}")
                results[index] = {
                    "query_id": queries[index]["id"],
                    "query": queries[index]["query"],
                    "category": queries[index]["category"],
                    "success": False,
                    "error": str(e),
                    "execution_time": 0
                }

    return results


def print_summary(results: List[Dict[str, Any]], total_time: float, workers: int):
    """Print execution summary."""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    judged_results = [r for r in results if r.get("judging", {}).get("judging_performed", False)]

    print(f"\n{'='*60}")
    print("BATCH EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total Queries: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total Time: {total_time:.1f} seconds (wall clock)")
    print(f"Cumulative Task Time: {sum(r['execution_time'] for r in results):.1f} seconds")
    if workers > 1:
        speedup = sum(r['execution_time'] for r in results) / total_time
        print(f"Speedup: {speedup:.1f}x")

    if judged_results:
        total_cost = sum(r["judging"]["judging_cost"]["total_cost"] for r in judged_results)
        avg_accuracy = sum(r["judging"]["accuracy_score"] for r in judged_results) / len(judged_results)
        avg_efficiency = sum(r["judging"]["efficiency_score"] for r in judged_results) / len(judged_results)
        print(f"\n📊 JUDGING SUMMARY:")
        print(f"  Judged Queries: {len(judged_results)}")
        print(f"  Average Accuracy Score: {avg_accuracy:.2f}/5")
        print(f"  Average Efficiency Score: {avg_efficiency:.2f}/5")
        print(f"  Total Judging Cost: ${total_cost:.4f}")

    if successful:
        print(f"\n✅ SUCCESSFUL QUERIES:")
        for r in successful:
            judge_info = ""
            if r.get("judging", {}).get("judging_performed", False):
                accuracy = r["judging"]["accuracy_score"]
                efficiency = r["judging"]["efficiency_score"]
                cost = r["judging"]["judging_cost"]["total_cost"]
                judge_info = f" | Acc: {accuracy}/5 | Eff: {efficiency}/5 | Cost: ${cost:.4f}"

            # Add detailed timing if available
            timing_info = ""
            if r.get("detailed_timing"):
                dt = r["detailed_timing"]
                tool_time = dt.get("total_tool_time", 0)
                thinking_time = dt.get("thinking_time", 0)
                efficiency = dt.get("tool_efficiency", 0)
                timing_info = f" | Tools: {tool_time:.1f}s | Think: {thinking_time:.1f}s | Eff: {efficiency:.1%}"

            print(f"  {r['query_id']}: {r['category']} ({r['execution_time']:.1f}s){timing_info}{judge_info}")

        # Show detailed tool breakdown for successful queries
        print(f"\n🔧 TOOL USAGE BREAKDOWN:")
        tool_aggregates = {}
        for r in successful:
            if r.get("detailed_timing", {}).get("tool_breakdown"):
                for tool_name, tool_stats in r["detailed_timing"]["tool_breakdown"].items():
                    if tool_name not in tool_aggregates:
                        tool_aggregates[tool_name] = {"count": 0, "total_duration": 0, "successes": 0, "failures": 0}

                    tool_aggregates[tool_name]["count"] += tool_stats["count"]
                    tool_aggregates[tool_name]["total_duration"] += tool_stats["total_duration"]
                    tool_aggregates[tool_name]["successes"] += tool_stats["successful_calls"]
                    tool_aggregates[tool_name]["failures"] += tool_stats["failed_calls"]

        for tool_name, stats in tool_aggregates.items():
            avg_duration = stats["total_duration"] / stats["count"] if stats["count"] > 0 else 0
            success_rate = stats["successes"] / (stats["successes"] + stats["failures"]) if (stats["successes"] + stats["failures"]) > 0 else 0
            print(f"  {tool_name}: {stats['count']} calls, {stats['total_duration']:.2f}s total, "
                  f"{avg_duration:.2f}s avg, {success_rate:.1%} success rate")

    if failed:
        print(f"\n❌ FAILED QUERIES:")
        for r in failed:
            error = r.get('error', 'Unknown error')
            print(f"  {r['query_id']}: {r['category']} - {error}")


def main():
    parser = argparse.ArgumentParser(description="Run batch queries using ReAct framework")
    parser.add_argument("queries_file", help="Path to queries JSON file")
    parser.add_argument("--count", type=int, default=5, help="Number of queries to run (default: 5)")
    parser.add_argument("--template", type=str, default="templates/data_analysis_agent_prompt.txt",
                       help="Template file path")
    parser.add_argument("--max-tools", type=int, default=15, help="Maximum tool calls per query")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per script execution")
    parser.add_argument("--model", type=str, default="anthropic:claude-sonnet-4-20250514",
                       help="Model to use, e.g. 'anthropic:claude-sonnet-4-20250514' or 'ollama:llama3.1'")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for results (default: batch_results_TIMESTAMP.json)")
    parser.add_argument("--no-judge", action="store_true", help="Disable LLM judging of results")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--no-stream", action="store_true", help="Disable live streaming of agent output")
    parser.add_argument("--no-console-updates", action="store_true", help="Disable tool progress updates from agent")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1 for serial execution)")

    args = parser.parse_args()

    # Load configuration
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

    # Load queries
    try:
        data = load_queries(args.queries_file)
        queries = data["queries"][:args.count]  # Take first N queries
    except Exception as e:
        print(f"Error loading queries file: {e}")
        sys.exit(1)

    # Set up output file in results/ folder
    if args.output is None:
        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/batch_results_{timestamp}.json"

    # Base arguments for run_agent.py
    base_args = [
        "--template", args.template,
        "--max-tools", str(args.max_tools),
        "--timeout", str(args.timeout),
        "--model", args.model
    ]

    # Default ON: pass console updates unless explicitly disabled
    if not args.no_console_updates:
        base_args.append("--console-updates")

    # Determine if judging should be enabled
    judging_enabled = config.get("judging", {}).get("enabled", True) and not args.no_judge

    print(f"Running {len(queries)} queries with ReAct framework...")
    print(f"Template: {args.template}")
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers} ({'parallel' if args.workers > 1 else 'serial'})")
    print(f"Judging: {'Enabled' if judging_enabled else 'Disabled'}")
    print(f"Results will be saved to: {args.output}")

    # Run queries (serial or parallel)
    start_time = time.time()
    results = run_queries_parallel(queries, base_args, config, judging_enabled, args.no_stream, args.workers)
    total_time = time.time() - start_time

    # Save and display results
    save_results(results, args.output)
    print_summary(results, total_time, args.workers)

    print(f"\nDetailed results saved to: {args.output}")

    if args.workers > 1:
        print(f"\n🚀 Parallel execution completed with {args.workers} workers in {total_time:.1f}s")


if __name__ == "__main__":
    main()
