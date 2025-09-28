"""Idea specification and validation"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime


@dataclass
class IdeaSpec:
    """Complete specification for an analysis idea"""

    # Core identification
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    query: str = ""  # Main research question

    # Context and inspiration
    event_context: Optional[str] = None  # Reference to event from inspiration
    technique_context: Optional[str] = None  # Reference to technique from inspiration

    # Analysis design
    method: str = "event_study"  # Primary analytical method
    design_type: str = "observational"  # "experimental", "observational", "descriptive"

    # Data scope
    assets: List[str] = field(default_factory=list)  # Specific pipelines/locations
    time_window: Dict[str, Any] = field(default_factory=dict)  # Start, end, pre/post periods
    filters: Dict[str, Any] = field(default_factory=dict)  # Additional data filters

    # Expected outputs
    expected_artifacts: List[str] = field(default_factory=list)  # Files/plots to generate
    trader_hook: str = ""  # Business relevance and decision support

    # Metadata
    generation: int = 0  # Evolution depth (0 = seed)
    parent_id: Optional[str] = None  # Parent idea if evolved
    confidence: float = 0.5  # Confidence in executability
    complexity: int = 3  # 1-5 scale of implementation difficulty

    # Scoring
    scores: Dict[str, float] = field(default_factory=dict)

    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate the specification after creation"""
        if not self.query:
            raise ValueError("Query is required")

        if not self.assets:
            self.confidence *= 0.8  # Reduce confidence for vague asset specification

        if not self.trader_hook:
            self.confidence *= 0.7  # Reduce confidence without business relevance

    def to_query_text(self) -> str:
        """Convert to natural language query for execution"""
        query_text = self.query

        # Add context if available
        if self.event_context:
            query_text += f" Context: {self.event_context}."

        # Add method guidance
        if self.method and self.method != "descriptive":
            query_text += f" Use {self.method} methodology."

        # Add asset focus
        if self.assets:
            assets_str = ", ".join(self.assets[:3])  # Limit to first 3
            query_text += f" Focus on: {assets_str}."

        # Add time window
        if self.time_window:
            if "start" in self.time_window and "end" in self.time_window:
                query_text += f" Time period: {self.time_window['start']} to {self.time_window['end']}."

        return query_text

    def estimate_execution_cost(self) -> Dict[str, float]:
        """Estimate tokens and time for execution"""
        base_tokens = 8000

        # Adjust for complexity
        complexity_multiplier = 1 + (self.complexity - 3) * 0.3

        # Adjust for method type
        method_multipliers = {
            "descriptive": 0.7,
            "event_study": 1.2,
            "did": 1.3,
            "synthetic_control": 1.5,
            "var": 1.4,
            "anomaly_detection": 1.1
        }

        method_mult = method_multipliers.get(self.method, 1.0)

        estimated_tokens = int(base_tokens * complexity_multiplier * method_mult)
        estimated_minutes = estimated_tokens / 500  # Rough estimate

        return {
            "tokens": estimated_tokens,
            "minutes": estimated_minutes,
            "cost_usd": estimated_tokens * 0.000003  # Rough cost estimate
        }

    def validate_executability(self) -> tuple[bool, List[str]]:
        """Check if idea can be executed"""
        issues = []

        if not self.query:
            issues.append("Missing query")

        if len(self.query) < 10:
            issues.append("Query too short")

        if self.complexity > 4:
            issues.append("Complexity too high for MVP")

        if not self.trader_hook:
            issues.append("Missing business relevance")

        # Check cost estimate
        cost = self.estimate_execution_cost()
        if cost["tokens"] > 20000:
            issues.append("Estimated cost too high")

        return len(issues) == 0, issues

    def add_score(self, dimension: str, score: float, max_score: float = 1.0):
        """Add a score for a specific dimension"""
        self.scores[dimension] = min(score, max_score)

    def get_overall_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted overall score"""
        if not self.scores:
            return 0.0

        default_weights = {
            "domain_relevance": 0.3,
            "trader_value": 0.3,
            "technical_rigor": 0.2,
            "novelty": 0.2
        }

        weights = weights or default_weights

        total_score = 0.0
        total_weight = 0.0

        for dimension, weight in weights.items():
            if dimension in self.scores:
                total_score += self.scores[dimension] * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0