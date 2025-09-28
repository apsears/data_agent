"""Seed generation from inspiration libraries"""

import random
from typing import List, Dict, Any
from .specifications import IdeaSpec
from ..inspiration.events import EventLibrary
from ..inspiration.techniques import TechniqueLibrary


class SeedGenerator:
    """Generate initial ideas from domain inspiration"""

    def __init__(self, events: EventLibrary, techniques: TechniqueLibrary):
        self.events = events
        self.techniques = techniques

    def generate_all_seeds(self, config: Dict[str, Any]) -> List[IdeaSpec]:
        """Generate complete set of seed ideas"""
        seeds = []

        # Event-driven seeds
        seeds.extend(self.generate_event_driven_seeds(
            config.get("n_event_driven", 5)
        ))

        # Technique-driven seeds
        seeds.extend(self.generate_technique_driven_seeds(
            config.get("n_technique_driven", 5)
        ))

        # Hybrid seeds (event + technique)
        seeds.extend(self.generate_hybrid_seeds(
            config.get("n_hybrid", 5)
        ))

        # Trader question seeds
        seeds.extend(self.generate_trader_question_seeds(
            config.get("n_trader_questions", 5)
        ))

        return seeds

    def generate_event_driven_seeds(self, n: int) -> List[IdeaSpec]:
        """Generate ideas focused on major events"""
        seeds = []

        for event in self.events.events[:n]:
            analysis_suggestion = self.events.suggest_analysis_for_event(event)

            seed = IdeaSpec(
                query=analysis_suggestion["query"],
                event_context=f"{event.name}: {event.description}",
                method=analysis_suggestion["method"],
                design_type="causal",
                assets=event.affected_assets,
                time_window={
                    "start": event.start_date,
                    "end": event.end_date,
                    "window_type": analysis_suggestion["window"]
                },
                expected_artifacts=[
                    "pretrend_plot.png",
                    "effect_estimates.json",
                    "robustness_checks.json"
                ],
                trader_hook=f"Impact on {event.affected_assets[0]} flows affects locational spreads and capacity constraints. {analysis_suggestion['expected']}",
                complexity=3 if event.event_type == "weather" else 4,
                confidence=0.8
            )

            seeds.append(seed)

        return seeds

    def generate_technique_driven_seeds(self, n: int) -> List[IdeaSpec]:
        """Generate ideas focused on analytical techniques"""
        seeds = []

        techniques_to_use = random.sample(self.techniques.techniques, min(n, len(self.techniques.techniques)))

        for technique in techniques_to_use:
            # Generate a query based on the technique
            query = self._create_technique_query(technique)

            seed = IdeaSpec(
                query=query,
                technique_context=f"{technique.name}: {technique.description}",
                method=technique.name.lower().replace(" ", "_"),
                design_type=technique.method_type,
                assets=["Major interstate pipelines"],  # Generic for technique-focused
                time_window={"lookback_months": 12},
                expected_artifacts=technique.output_artifacts,
                trader_hook=technique.trader_value,
                complexity=3,
                confidence=0.7
            )

            seeds.append(seed)

        return seeds

    def generate_hybrid_seeds(self, n: int) -> List[IdeaSpec]:
        """Generate ideas combining events and techniques"""
        seeds = []

        for i in range(n):
            # Randomly pair event and technique
            event = random.choice(self.events.events)
            technique = random.choice(self.techniques.techniques)

            query = f"Using {technique.name.lower()}, analyze how {event.name} affected {random.choice(event.affected_assets)} operations"

            seed = IdeaSpec(
                query=query,
                event_context=f"{event.name}: {event.description}",
                technique_context=f"{technique.name}: {technique.description}",
                method=technique.name.lower().replace(" ", "_"),
                design_type="causal",
                assets=event.affected_assets,
                time_window={
                    "event_date": event.start_date,
                    "pre_period": 60,
                    "post_period": 90
                },
                expected_artifacts=technique.output_artifacts + ["event_study.png"],
                trader_hook=f"Combines {event.expected_impact} with {technique.trader_value}",
                complexity=4,
                confidence=0.6
            )

            seeds.append(seed)

        return seeds

    def generate_trader_question_seeds(self, n: int) -> List[IdeaSpec]:
        """Generate ideas from trader questions"""
        seeds = []

        for i, trader_q in enumerate(self.techniques.trader_questions[:n]):
            technique = self.techniques.suggest_technique_for_question(trader_q["question"])

            seed = IdeaSpec(
                query=trader_q["question"],
                technique_context=f"Trader priority: {trader_q['value']}",
                method=trader_q["method"],
                design_type="observational",
                assets=["All major pipelines"],
                time_window={"lookback_months": 18},
                expected_artifacts=[
                    "analysis_results.json",
                    "trader_summary.md",
                    "decision_metrics.csv"
                ],
                trader_hook=trader_q["value"],
                complexity=2,
                confidence=0.9
            )

            seeds.append(seed)

        return seeds

    def _create_technique_query(self, technique) -> str:
        """Create a query based on a technique"""
        technique_queries = {
            "Event Study DiD": "What was the causal impact of recent infrastructure changes on flow patterns?",
            "Regression Discontinuity in Time": "Where do capacity thresholds create discontinuous flow responses?",
            "Synthetic Control": "How would flows have evolved without the major capacity addition?",
            "Instrumental Variables": "What are the causal elasticities of flow responses to upstream shocks?",
            "Bayesian Structural Time Series": "What was the probabilistic impact of infrastructure changes?",
            "Panel VAR Lead-Lag": "Which pipelines show predictive relationships in flow changes?",
            "Transfer Entropy": "Which interconnects exhibit nonlinear lead-lag flow relationships?",
            "Network Flow Centrality": "What pipeline locations act as critical bottlenecks?",
            "Utilization Constraint Detection": "Where and when do pipeline capacity constraints bind most frequently?",
            "Mass Balance Tightness": "How do pipeline utilization levels affect delivery flexibility?",
            "Causal Forests": "Which customer types benefit most from capacity changes?",
            "Quantile DiD": "How do infrastructure changes affect flow distribution tails?",
            "Balance Violation Detection": "What days show unusual receipt-delivery imbalances by pipeline?",
            "Zero-Inflation Regime Analysis": "How has the frequency of zero-flow days changed over time?",
            "Duplicate Impact Analysis": "How do data quality issues affect analysis conclusions?",
            "Hidden Segment Clustering": "What operational archetypes exist in pipeline behavior?",
            "Conditional Dependency Mapping": "What sparse conditional associations exist between locations?",
            "Counterfactual Flow Forecasting": "How would flows change under alternative shock scenarios?",
            "Optimal Transport Reallocation": "Where do constrained flows get rerouted during capacity limits?"
        }

        return technique_queries.get(technique.name, f"Analyze pipeline data using {technique.name.lower()}")