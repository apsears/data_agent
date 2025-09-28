"""Evolution operators for improving ideas"""

import random
import copy
from typing import List, Dict, Any
from ..core.specifications import IdeaSpec
from ..inspiration.events import EventLibrary
from ..inspiration.techniques import TechniqueLibrary


class EvolutionOperators:
    """Simple evolution operators for MVP"""

    def __init__(self, events: EventLibrary, techniques: TechniqueLibrary):
        self.events = events
        self.techniques = techniques

    def make_more_specific(self, idea: IdeaSpec) -> IdeaSpec:
        """Narrow focus to specific assets, time periods, or conditions"""
        evolved = copy.deepcopy(idea)
        evolved.parent_id = idea.id
        evolved.generation = idea.generation + 1

        # Make assets more specific
        if "major" in str(evolved.assets).lower() or "all" in str(evolved.assets).lower():
            # Replace generic with specific pipelines
            specific_assets = [
                "Gulf South Pipeline Company, LP",
                "Texas Eastern Transmission, LP",
                "Transcontinental Gas Pipe Line Company, LLC"
            ]
            evolved.assets = random.sample(specific_assets, min(2, len(specific_assets)))

        # Make time window more specific
        if not evolved.time_window or "lookback_months" in evolved.time_window:
            evolved.time_window = {
                "start": "2023-01-01",
                "end": "2024-12-31",
                "focus": "high_volatility_periods"
            }

        # Add specificity to query
        if "where" not in evolved.query.lower() and "which" not in evolved.query.lower():
            evolved.query = evolved.query.replace("pipeline", "interstate pipeline in Texas/Louisiana")

        # Add specific conditions
        evolved.filters = {
            "min_flow_threshold": 1000,  # MCF/day
            "exclude_weekends": True,
            "exclude_holidays": True
        }

        evolved.trader_hook += " Focus on high-volume corridors with predictable patterns."
        evolved.confidence *= 1.1  # More specific is generally better
        evolved.complexity += 1

        return evolved

    def add_comparison(self, idea: IdeaSpec) -> IdeaSpec:
        """Add control groups or baseline comparisons"""
        evolved = copy.deepcopy(idea)
        evolved.parent_id = idea.id
        evolved.generation = idea.generation + 1

        # Modify query to include comparison
        if "compared to" not in evolved.query.lower() and "versus" not in evolved.query.lower():
            if evolved.event_context:
                evolved.query += " How does this compare to unaffected regions?"
            else:
                evolved.query += " How does this compare to historical patterns?"

        # Add control group suggestions
        if evolved.method in ["event_study", "did"]:
            evolved.filters["control_group"] = "neighboring_states"
            evolved.expected_artifacts.append("control_group_validation.png")

        # Add baseline comparison
        evolved.expected_artifacts.extend([
            "baseline_comparison.json",
            "difference_analysis.png"
        ])

        # Update trader hook
        evolved.trader_hook += " Comparison provides confidence bounds for trading decisions."
        evolved.complexity += 1
        evolved.confidence *= 1.05

        return evolved

    def add_robustness(self, idea: IdeaSpec) -> IdeaSpec:
        """Add statistical robustness checks and validation"""
        evolved = copy.deepcopy(idea)
        evolved.parent_id = idea.id
        evolved.generation = idea.generation + 1

        # Add robustness language to query
        if "robust" not in evolved.query.lower():
            evolved.query += " Include robustness checks and sensitivity analysis."

        # Add robustness artifacts
        robustness_artifacts = [
            "pretrend_test.png",
            "placebo_test.json",
            "sensitivity_analysis.png",
            "bootstrap_confidence_intervals.json"
        ]

        evolved.expected_artifacts.extend(robustness_artifacts)

        # Update method with robustness
        if evolved.method == "event_study":
            evolved.method = "robust_event_study"
        elif evolved.method == "did":
            evolved.method = "robust_did"

        # Add robustness filters
        evolved.filters.update({
            "require_pretrend_test": True,
            "bootstrap_iterations": 1000,
            "significance_level": 0.05,
            "cluster_standard_errors": "pipeline"
        })

        evolved.trader_hook += " Robustness checks provide confidence for risk management decisions."
        evolved.complexity += 2
        evolved.confidence *= 1.2

        return evolved

    def add_trader_angle(self, idea: IdeaSpec, inspiration: Dict[str, Any]) -> IdeaSpec:
        """Enhance business relevance and actionability"""
        evolved = copy.deepcopy(idea)
        evolved.parent_id = idea.id
        evolved.generation = idea.generation + 1

        # Add trader-specific question
        trader_angles = [
            "What are the implications for basis spreads?",
            "How does this affect capacity allocation decisions?",
            "What early warning signals does this provide?",
            "How can this inform hedging strategies?",
            "What risk management implications arise?"
        ]

        evolved.query += f" {random.choice(trader_angles)}"

        # Add trader-specific artifacts
        trader_artifacts = [
            "basis_impact_analysis.json",
            "trading_signals.csv",
            "risk_metrics.json",
            "decision_thresholds.yaml"
        ]

        evolved.expected_artifacts.extend(trader_artifacts[:2])  # Add 2 random ones

        # Enhance trader hook with specific actions
        action_suggestions = [
            "Monitor for 3+ day constraint periods as early warning",
            "Use flow changes >15% as basis spread indicators",
            "Apply 48-hour lead time for capacity reallocation",
            "Set alert thresholds at 90th percentile utilization"
        ]

        evolved.trader_hook += f" Actionable insight: {random.choice(action_suggestions)}"
        evolved.confidence *= 1.15

        return evolved

    def cross_pollinate(self, idea_a: IdeaSpec, idea_b: IdeaSpec) -> IdeaSpec:
        """Combine elements from two different ideas"""
        # Use idea_a as base
        evolved = copy.deepcopy(idea_a)
        evolved.parent_id = f"{idea_a.id}+{idea_b.id}"
        evolved.generation = max(idea_a.generation, idea_b.generation) + 1

        # Take method from idea_b if it's more sophisticated
        method_complexity = {
            "descriptive": 1,
            "constraint_detection": 2,
            "event_study": 3,
            "did": 4,
            "synthetic_control": 5,
            "var": 4
        }

        if method_complexity.get(idea_b.method, 0) > method_complexity.get(idea_a.method, 0):
            evolved.method = idea_b.method

        # Combine assets
        combined_assets = list(set(idea_a.assets + idea_b.assets))
        evolved.assets = combined_assets[:4]  # Limit to 4 assets

        # Combine contexts
        if idea_b.event_context and not evolved.event_context:
            evolved.event_context = idea_b.event_context
        elif idea_b.technique_context and not evolved.technique_context:
            evolved.technique_context = idea_b.technique_context

        # Merge trader hooks
        evolved.trader_hook = f"{evolved.trader_hook} Additionally: {idea_b.trader_hook[:100]}..."

        # Take best confidence
        evolved.confidence = max(idea_a.confidence, idea_b.confidence) * 0.9  # Slight penalty for complexity

        # Update query to reflect combination
        evolved.query = f"Combined analysis: {idea_a.query[:60]}... and {idea_b.query[:60]}..."

        evolved.complexity = min(5, max(idea_a.complexity, idea_b.complexity) + 1)

        return evolved

    def evolve_idea(self, idea: IdeaSpec, operators: List[str] = None) -> List[IdeaSpec]:
        """Apply multiple evolution operators to an idea"""
        if operators is None:
            operators = ["make_more_specific", "add_comparison", "add_robustness", "add_trader_angle"]

        evolutions = []

        for op_name in operators:
            try:
                if hasattr(self, op_name):
                    operator = getattr(self, op_name)
                    if op_name == "add_trader_angle":
                        evolved = operator(idea, {})  # Empty inspiration dict for MVP
                    else:
                        evolved = operator(idea)

                    # Validate the evolved idea
                    valid, issues = evolved.validate_executability()
                    if valid:
                        evolutions.append(evolved)
                    else:
                        # Try to fix simple issues
                        if "Missing business relevance" in issues and not evolved.trader_hook:
                            evolved.trader_hook = "Provides operational insights for capacity planning."
                            valid, _ = evolved.validate_executability()
                            if valid:
                                evolutions.append(evolved)

            except Exception as e:
                # Skip failed evolutions in MVP
                print(f"Evolution operator {op_name} failed: {e}")
                continue

        return evolutions