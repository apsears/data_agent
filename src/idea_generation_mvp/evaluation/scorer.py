"""Scoring and selection for generated ideas"""

import json
import re
from typing import List, Dict, Any, Tuple
from collections import Counter
from ..core.specifications import IdeaSpec


class IdeaScorer:
    """Score ideas on multiple dimensions"""

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "domain_relevance": 0.3,
            "trader_value": 0.3,
            "technical_rigor": 0.2,
            "novelty": 0.2
        }
        self.previous_ideas = []  # For novelty calculation

    def score_idea(self, spec: IdeaSpec, execution_result: Dict[str, Any] = None) -> Dict[str, float]:
        """Score an idea across all dimensions"""
        scores = {}

        # Domain relevance - how well does it match inspiration
        scores["domain_relevance"] = self._score_domain_relevance(spec)

        # Trader value - business relevance and actionability
        scores["trader_value"] = self._score_trader_value(spec)

        # Technical rigor - methodological soundness
        scores["technical_rigor"] = self._score_technical_rigor(spec, execution_result)

        # Novelty - uniqueness compared to previous ideas
        scores["novelty"] = self._score_novelty(spec)

        # Overall score
        scores["overall"] = sum(scores[dim] * self.weights[dim] for dim in self.weights.keys())

        # Store for novelty calculation
        self.previous_ideas.append(spec)

        return scores

    def _score_domain_relevance(self, spec: IdeaSpec) -> float:
        """Score how well the idea leverages domain knowledge"""
        score = 0.0

        # Event context bonus
        if spec.event_context:
            score += 0.4
            # Major events get higher scores
            major_keywords = ["freeport", "matterhorn", "mvp", "storm", "outage", "expansion"]
            if any(keyword in spec.event_context.lower() for keyword in major_keywords):
                score += 0.2

        # Technique context bonus
        if spec.technique_context:
            score += 0.3

        # Specific assets vs generic
        if spec.assets and "major" not in str(spec.assets).lower():
            score += 0.2

        # Method sophistication
        method_scores = {
            "synthetic_control": 0.3,
            "did": 0.25,
            "event_study": 0.2,
            "constraint_detection": 0.15,
            "descriptive": 0.05
        }
        score += method_scores.get(spec.method, 0.1)

        return min(1.0, score)

    def _score_trader_value(self, spec: IdeaSpec) -> float:
        """Score business relevance and actionability"""
        score = 0.0

        # Trader hook quality
        if spec.trader_hook:
            hook_lower = spec.trader_hook.lower()

            # Action-oriented language
            action_words = ["threshold", "alert", "signal", "hedge", "allocate", "risk", "spread", "basis"]
            action_count = sum(1 for word in action_words if word in hook_lower)
            score += min(0.4, action_count * 0.1)

            # Specific numbers/timing
            if re.search(r'\d+', spec.trader_hook):
                score += 0.2

            # Decision support language
            decision_words = ["decision", "trading", "capacity", "constraint", "opportunity"]
            if any(word in hook_lower for word in decision_words):
                score += 0.2

        # Expected artifacts relevance
        if spec.expected_artifacts:
            trader_artifacts = ["alert", "signal", "threshold", "basis", "risk", "decision"]
            artifact_score = sum(1 for artifact in spec.expected_artifacts
                               if any(word in artifact.lower() for word in trader_artifacts))
            score += min(0.3, artifact_score * 0.1)

        # Complexity appropriateness (not too complex for trading use)
        if spec.complexity <= 3:
            score += 0.1

        return min(1.0, score)

    def _score_technical_rigor(self, spec: IdeaSpec, execution_result: Dict[str, Any] = None) -> float:
        """Score methodological soundness"""
        score = 0.0

        # Method appropriateness
        causal_methods = ["event_study", "did", "synthetic_control", "var"]
        if spec.method in causal_methods and spec.design_type == "causal":
            score += 0.3

        # Robustness artifacts
        if spec.expected_artifacts:
            rigor_artifacts = ["pretrend", "placebo", "robust", "bootstrap", "confidence"]
            rigor_count = sum(1 for artifact in spec.expected_artifacts
                            if any(word in artifact.lower() for word in rigor_artifacts))
            score += min(0.3, rigor_count * 0.1)

        # Time window appropriateness
        if spec.time_window:
            if "start" in spec.time_window and "end" in spec.time_window:
                score += 0.2

        # Execution success bonus
        if execution_result and execution_result.get("success"):
            score += 0.3

            # Look for statistical terms in output
            if "stdout" in execution_result:
                output = execution_result["stdout"].lower()
                stats_terms = ["confidence", "significance", "p-value", "robust", "test"]
                if any(term in output for term in stats_terms):
                    score += 0.2

        # Confidence penalty for low confidence ideas
        score *= spec.confidence

        return min(1.0, score)

    def _score_novelty(self, spec: IdeaSpec) -> float:
        """Score uniqueness compared to previous ideas"""
        if not self.previous_ideas:
            return 1.0

        # Compare to previous ideas
        similarities = []
        for prev_spec in self.previous_ideas:
            similarity = self._calculate_similarity(spec, prev_spec)
            similarities.append(similarity)

        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities) if similarities else 0
        novelty = 1.0 - max_similarity

        # Bonus for new methods or event combinations
        method_counts = Counter(prev.method for prev in self.previous_ideas)
        if spec.method not in method_counts:
            novelty += 0.2

        return min(1.0, novelty)

    def _calculate_similarity(self, spec1: IdeaSpec, spec2: IdeaSpec) -> float:
        """Calculate similarity between two specs"""
        similarity = 0.0

        # Method similarity
        if spec1.method == spec2.method:
            similarity += 0.4

        # Asset overlap
        if spec1.assets and spec2.assets:
            overlap = len(set(spec1.assets) & set(spec2.assets))
            total = len(set(spec1.assets) | set(spec2.assets))
            if total > 0:
                similarity += 0.3 * (overlap / total)

        # Event context similarity
        if spec1.event_context and spec2.event_context:
            if spec1.event_context == spec2.event_context:
                similarity += 0.2

        # Query similarity (simple word overlap)
        if spec1.query and spec2.query:
            words1 = set(spec1.query.lower().split())
            words2 = set(spec2.query.lower().split())
            overlap = len(words1 & words2)
            total = len(words1 | words2)
            if total > 0:
                similarity += 0.1 * (overlap / total)

        return similarity


class PortfolioSelector:
    """Select diverse, high-value portfolios"""

    def __init__(self, diversity_weight: float = 0.3):
        self.diversity_weight = diversity_weight

    def select_portfolio(self,
                        specs: List[IdeaSpec],
                        scores: Dict[str, Dict[str, float]],
                        n: int = 5) -> List[IdeaSpec]:
        """Select a diverse portfolio of high-scoring ideas"""

        if len(specs) <= n:
            return specs

        # Start with highest scoring idea
        scored_specs = [(spec, scores.get(spec.id, {}).get("overall", 0)) for spec in specs]
        scored_specs.sort(key=lambda x: x[1], reverse=True)

        portfolio = [scored_specs[0][0]]  # Start with best idea
        remaining = [spec for spec, _ in scored_specs[1:]]

        while len(portfolio) < n and remaining:
            best_candidate = None
            best_value = -1

            for candidate in remaining:
                # Base score
                base_score = scores.get(candidate.id, {}).get("overall", 0)

                # Diversity bonus
                diversity_score = self._calculate_diversity_bonus(candidate, portfolio)

                # Combined value
                total_value = (1 - self.diversity_weight) * base_score + self.diversity_weight * diversity_score

                if total_value > best_value:
                    best_value = total_value
                    best_candidate = candidate

            if best_candidate:
                portfolio.append(best_candidate)
                remaining.remove(best_candidate)

        return portfolio

    def _calculate_diversity_bonus(self, candidate: IdeaSpec, portfolio: List[IdeaSpec]) -> float:
        """Calculate diversity bonus for a candidate"""
        if not portfolio:
            return 1.0

        # Check method diversity
        portfolio_methods = set(spec.method for spec in portfolio)
        method_bonus = 0.3 if candidate.method not in portfolio_methods else 0

        # Check event diversity
        portfolio_events = set(spec.event_context for spec in portfolio if spec.event_context)
        event_bonus = 0.3 if candidate.event_context not in portfolio_events else 0

        # Check asset diversity
        portfolio_assets = set()
        for spec in portfolio:
            portfolio_assets.update(spec.assets)

        candidate_assets = set(candidate.assets)
        asset_overlap = len(candidate_assets & portfolio_assets) / len(candidate_assets) if candidate_assets else 0
        asset_bonus = 0.4 * (1 - asset_overlap)

        return method_bonus + event_bonus + asset_bonus