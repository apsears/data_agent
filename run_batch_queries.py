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


def run_single_query(query: Dict[str, Any], base_args: List[str]) -> Dict[str, Any]:
    """Run a single query using run_agent.py and capture results."""
    print(f"\n{'='*60}")
    print(f"Running Query {query['id']}: {query['category']}")
    print(f"Query: {query['query']}")
    print(f"{'='*60}")

    # Build command
    cmd = [
        sys.executable, "run_agent.py",
        "--task", query["query"],
        *base_args
    ]

    start_time = time.time()

    try:
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per query
        )

        execution_time = time.time() - start_time

        # Parse the output to extract final answer
        stdout_lines = result.stdout.strip().split('\n')
        final_answer_start = None

        for i, line in enumerate(stdout_lines):
            if "=== FINAL ANSWER ===" in line:
                final_answer_start = i + 1
                break

        final_answer = ""
        if final_answer_start:
            # Join lines from final answer section until end
            final_answer = '\n'.join(stdout_lines[final_answer_start:])

        # Extract run directory for artifacts
        run_directory = None
        for line in stdout_lines:
            if "Run directory:" in line:
                run_directory = line.split("Run directory:")[-1].strip()
                break

        return {
            "query_id": query["id"],
            "query": query["query"],
            "category": query["category"],
            "success": result.returncode == 0,
            "execution_time": execution_time,
            "final_answer": final_answer,
            "run_directory": run_directory,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }

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


def _load_judging_template(config: Dict[str, Any]) -> Template:
    """Load and parse the judging template."""
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
            return json.dumps(tool_use.input), response.usage.output_tokens
        else:
            return response.content[0].text, response.usage.output_tokens

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


def _parse_judging_response(judge_response: str) -> Dict[str, Any]:
    """Parse JSON from structured judging response."""
    # With structured output, response should be clean JSON
    try:
        return json.loads(judge_response)
    except json.JSONDecodeError:
        # Fallback for any edge cases
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
    """Judge a single query result using LLM evaluation."""
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

        # Render judging prompt
        judging_prompt = template.render(
            query=query_data["query"],
            expected_answer=query_data.get("expected_answer", "No reference answer provided"),
            actual_answer=actual_answer
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
            "explanation": judging_data.get("explanation", "No explanation provided"),
            "numerical_comparison": judging_data.get("numerical_comparison", ""),
            "methodology_assessment": judging_data.get("methodology_assessment", ""),
            "completeness_check": judging_data.get("completeness_check", ""),
            "confidence": judging_data.get("confidence", 0.0),
            "judging_cost": judging_cost,
            "raw_judging_output": judge_response
        }

    except Exception as e:
        return {
            "judging_performed": False,
            "error": f"Judging failed: {str(e)}",
            "judging_cost": {"input_tokens": 0, "output_tokens": 0, "total_cost": 0.0}
        }


def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save batch results to JSON file."""
    batch_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": len(results),
        "successful_queries": len([r for r in results if r["success"]]),
        "failed_queries": len([r for r in results if not r["success"]]),
        "total_execution_time": sum(r["execution_time"] for r in results),
        "results": results
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

    with open(output_file, 'w') as f:
        json.dump(batch_summary, f, indent=2)


def print_summary(results: List[Dict[str, Any]]):
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
    print(f"Total Time: {sum(r['execution_time'] for r in results):.1f} seconds")

    if judged_results:
        total_cost = sum(r["judging"]["judging_cost"]["total_cost"] for r in judged_results)
        avg_score = sum(r["judging"]["accuracy_score"] for r in judged_results) / len(judged_results)
        print(f"\n📊 JUDGING SUMMARY:")
        print(f"  Judged Queries: {len(judged_results)}")
        print(f"  Average Accuracy Score: {avg_score:.2f}/5")
        print(f"  Total Judging Cost: ${total_cost:.4f}")

    if successful:
        print(f"\n✅ SUCCESSFUL QUERIES:")
        for r in successful:
            judge_info = ""
            if r.get("judging", {}).get("judging_performed", False):
                score = r["judging"]["accuracy_score"]
                cost = r["judging"]["judging_cost"]["total_cost"]
                judge_info = f" | Score: {score}/5 | Cost: ${cost:.4f}"
            print(f"  {r['query_id']}: {r['category']} ({r['execution_time']:.1f}s){judge_info}")

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
                       help="Model to use")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for results (default: batch_results_TIMESTAMP.json)")
    parser.add_argument("--no-judge", action="store_true", help="Disable LLM judging of results")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")

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

    # Set up output file
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"batch_results_{timestamp}.json"

    # Base arguments for run_agent.py
    base_args = [
        "--template", args.template,
        "--max-tools", str(args.max_tools),
        "--timeout", str(args.timeout),
        "--model", args.model
    ]

    # Determine if judging should be enabled
    judging_enabled = config.get("judging", {}).get("enabled", True) and not args.no_judge

    print(f"Running {len(queries)} queries with ReAct framework...")
    print(f"Template: {args.template}")
    print(f"Model: {args.model}")
    print(f"Judging: {'Enabled' if judging_enabled else 'Disabled'}")
    print(f"Results will be saved to: {args.output}")

    # Run queries
    results = []
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Processing query {query['id']}...")
        result = run_single_query(query, base_args)

        # Add judging if enabled
        if judging_enabled and result["success"]:
            print(f"  Judging result...")
            judging_result = judge_single_result(result, query, config)
            result["judging"] = judging_result

        results.append(result)

        # Brief pause between queries
        if i < len(queries):
            time.sleep(2)

    # Save and display results
    save_results(results, args.output)
    print_summary(results)

    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()