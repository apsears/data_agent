#!/usr/bin/env python3
"""
Script to judge previously completed batch results.
This allows testing the judging system on existing query results without re-running queries.
"""

import argparse
import json
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Import judging functions from run_batch_queries.py
from run_batch_queries import (
    judge_single_result,
    load_queries,
    _load_judging_template,
    _extract_actual_answer,
    _call_judging_llm,
    _parse_judging_response,
    _calculate_judging_cost
)


def load_batch_results(results_file: str) -> Dict[str, Any]:
    """Load batch results from JSON file."""
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results file: {e}")
        sys.exit(1)


def judge_batch_results(batch_data: Dict[str, Any], queries_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Judge all results in a batch."""
    results = batch_data["results"]
    queries_by_id = {q["id"]: q for q in queries_data["queries"]}

    print(f"Judging {len(results)} results...")

    judged_results = []
    successful_judgments = 0
    failed_judgments = 0

    for i, result in enumerate(results, 1):
        query_id = result["query_id"]
        print(f"[{i}/{len(results)}] Judging query {query_id}...")

        # Find corresponding query data
        query_data = queries_by_id.get(query_id)
        if not query_data:
            print(f"  Warning: No query data found for {query_id}")
            result["judging"] = {
                "judging_performed": False,
                "error": "Query data not found",
                "judging_cost": {"input_tokens": 0, "output_tokens": 0, "total_cost": 0.0}
            }
            judged_results.append(result)
            failed_judgments += 1
            continue

        # Only judge successful queries
        if not result["success"]:
            print(f"  Skipping failed query {query_id}")
            result["judging"] = {
                "judging_performed": False,
                "accuracy_score": 0,
                "explanation": "Query execution failed, cannot evaluate",
                "judging_cost": {"input_tokens": 0, "output_tokens": 0, "total_cost": 0.0}
            }
            judged_results.append(result)
            failed_judgments += 1
            continue

        # Perform judging
        try:
            judging_result = judge_single_result(result, query_data, config)
            result["judging"] = judging_result

            if judging_result["judging_performed"]:
                score = judging_result.get("accuracy_score", -1)
                cost = judging_result["judging_cost"]["total_cost"]
                print(f"  Score: {score}/5 | Cost: ${cost:.4f}")
                successful_judgments += 1
            else:
                print(f"  Failed: {judging_result.get('error', 'Unknown error')}")
                failed_judgments += 1

        except Exception as e:
            print(f"  Error judging {query_id}: {e}")
            result["judging"] = {
                "judging_performed": False,
                "error": f"Judging failed: {str(e)}",
                "judging_cost": {"input_tokens": 0, "output_tokens": 0, "total_cost": 0.0}
            }
            failed_judgments += 1

        judged_results.append(result)

    # Update batch data with judging results
    judged_batch_data = batch_data.copy()
    judged_batch_data["results"] = judged_results

    # Add judging summary
    judged_successful = [r for r in judged_results if r.get("judging", {}).get("judging_performed", False)]
    if judged_successful:
        total_cost = sum(r["judging"]["judging_cost"]["total_cost"] for r in judged_successful)
        avg_score = sum(r["judging"]["accuracy_score"] for r in judged_successful) / len(judged_successful)

        judged_batch_data["judging_summary"] = {
            "total_judged": len(judged_successful),
            "successful_judgments": successful_judgments,
            "failed_judgments": failed_judgments,
            "average_accuracy_score": avg_score,
            "total_judging_cost": total_cost,
            "total_input_tokens": sum(r["judging"]["judging_cost"]["input_tokens"] for r in judged_successful),
            "total_output_tokens": sum(r["judging"]["judging_cost"]["output_tokens"] for r in judged_successful),
            "judging_timestamp": datetime.now().isoformat()
        }

    return judged_batch_data


def print_judging_summary(batch_data: Dict[str, Any]):
    """Print summary of judging results."""
    results = batch_data["results"]
    judging_summary = batch_data.get("judging_summary", {})

    judged_results = [r for r in results if r.get("judging", {}).get("judging_performed", False)]
    failed_results = [r for r in results if r.get("judging", {}).get("judging_performed", False) == False]

    print(f"\n{'='*60}")
    print("JUDGING SUMMARY")
    print(f"{'='*60}")
    print(f"Total Results: {len(results)}")
    print(f"Successfully Judged: {len(judged_results)}")
    print(f"Failed to Judge: {len(failed_results)}")

    if judging_summary:
        print(f"Average Accuracy Score: {judging_summary['average_accuracy_score']:.2f}/5")
        print(f"Total Judging Cost: ${judging_summary['total_judging_cost']:.4f}")
        print(f"Total Tokens: {judging_summary['total_input_tokens']:,} input, {judging_summary['total_output_tokens']:,} output")

    if judged_results:
        print(f"\n✅ SUCCESSFULLY JUDGED:")
        for r in judged_results:
            score = r["judging"]["accuracy_score"]
            cost = r["judging"]["judging_cost"]["total_cost"]
            print(f"  {r['query_id']}: Score {score}/5 | ${cost:.4f}")

    if failed_results:
        print(f"\n❌ FAILED TO JUDGE:")
        for r in failed_results:
            error = r["judging"].get("error", "Unknown error")
            print(f"  {r['query_id']}: {error}")


def main():
    parser = argparse.ArgumentParser(description="Judge previously completed batch results")
    parser.add_argument("results_file", help="Path to batch results JSON file")
    parser.add_argument("queries_file", help="Path to queries JSON file")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for judged results (default: add _judged suffix to input file)")

    args = parser.parse_args()

    # Load configuration
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

    # Check if judging is enabled
    if not config.get("judging", {}).get("enabled", True):
        print("Judging is disabled in configuration")
        sys.exit(1)

    # Load batch results and queries
    batch_data = load_batch_results(args.results_file)
    queries_data = load_queries(args.queries_file)

    print(f"Loaded batch results from: {args.results_file}")
    print(f"Loaded {len(queries_data['queries'])} queries from: {args.queries_file}")
    print(f"Model: {config.get('judging', {}).get('model', 'not configured')}")

    # Judge the results
    judged_data = judge_batch_results(batch_data, queries_data, config)

    # Set up output file
    if args.output is None:
        results_path = Path(args.results_file)
        args.output = str(results_path.with_stem(f"{results_path.stem}_judged"))

    # Save judged results
    with open(args.output, 'w') as f:
        json.dump(judged_data, f, indent=2)

    # Print summary
    print_judging_summary(judged_data)
    print(f"\nJudged results saved to: {args.output}")


if __name__ == "__main__":
    main()