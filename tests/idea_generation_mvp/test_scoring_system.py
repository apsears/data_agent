"""Test scoring and portfolio selection functionality"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from idea_generation_mvp.core.specifications import IdeaSpec
from idea_generation_mvp.evaluation.scorer import IdeaScorer, PortfolioSelector


class TestIdeaScorer:
    """Test idea scoring functionality"""

    @pytest.fixture
    def scorer(self):
        return IdeaScorer()

    @pytest.fixture
    def custom_scorer(self):
        weights = {
            "domain_relevance": 0.4,
            "trader_value": 0.3,
            "technical_rigor": 0.2,
            "novelty": 0.1
        }
        return IdeaScorer(weights)

    @pytest.fixture
    def sample_idea(self):
        return IdeaSpec(
            query="What was the causal impact of Freeport LNG explosion on pipeline flows?",
            event_context="Freeport LNG explosion on June 8, 2022",
            technique_context="Event Study DiD: Causal impact measurement",
            method="event_study",
            assets=["Gulf South Pipeline", "Freeport LNG"],
            trader_hook="Impact analysis for basis trading decisions",
            expected_artifacts=["causal_estimate", "confidence_intervals", "pretrend_test"],
            confidence=0.8
        )

    def test_scorer_initializes(self, scorer):
        """Test that scorer initializes with default weights"""
        assert scorer is not None
        assert hasattr(scorer, 'weights')
        assert len(scorer.weights) == 4
        assert sum(scorer.weights.values()) == pytest.approx(1.0)

    def test_custom_weights(self, custom_scorer):
        """Test scorer with custom weights"""
        assert custom_scorer.weights["domain_relevance"] == 0.4
        assert custom_scorer.weights["trader_value"] == 0.3
        assert sum(custom_scorer.weights.values()) == pytest.approx(1.0)

    def test_score_idea_returns_valid_scores(self, scorer, sample_idea):
        """Test that scoring returns valid score dict"""
        scores = scorer.score_idea(sample_idea)

        assert isinstance(scores, dict)
        expected_keys = ["domain_relevance", "trader_value", "technical_rigor", "novelty", "overall"]

        for key in expected_keys:
            assert key in scores
            assert isinstance(scores[key], (int, float))
            assert 0.0 <= scores[key] <= 1.0

    def test_domain_relevance_scoring(self, scorer):
        """Test domain relevance scoring logic"""
        # High domain relevance idea
        high_domain_idea = IdeaSpec(
            query="Freeport LNG impact analysis",
            event_context="Freeport LNG explosion major event",
            technique_context="Synthetic Control technique",
            method="synthetic_control",
            assets=["Gulf South Pipeline", "Specific meters"]
        )

        # Low domain relevance idea
        low_domain_idea = IdeaSpec(
            query="Generic analysis",
            method="descriptive",
            assets=["Major pipelines"]
        )

        high_scores = scorer.score_idea(high_domain_idea)
        low_scores = scorer.score_idea(low_domain_idea)

        assert high_scores["domain_relevance"] > low_scores["domain_relevance"]

    def test_trader_value_scoring(self, scorer):
        """Test trader value scoring logic"""
        # High trader value idea
        high_trader_idea = IdeaSpec(
            query="Trading signal analysis",
            trader_hook="Provides trading signals for basis decisions with specific thresholds",
            expected_artifacts=["trading_signal", "risk_alert", "basis_threshold"]
        )

        # Low trader value idea
        low_trader_idea = IdeaSpec(
            query="Academic analysis",
            trader_hook="Interesting for research purposes",
            expected_artifacts=["academic_paper"]
        )

        high_scores = scorer.score_idea(high_trader_idea)
        low_scores = scorer.score_idea(low_trader_idea)

        assert high_scores["trader_value"] > low_scores["trader_value"]

    def test_technical_rigor_scoring(self, scorer):
        """Test technical rigor scoring logic"""
        # High rigor idea
        high_rigor_idea = IdeaSpec(
            query="Causal analysis",
            method="synthetic_control",
            design_type="causal",
            expected_artifacts=["pretrend_test", "placebo_test", "robust_standard_errors"],
            time_window={"start": "2022-01-01", "end": "2022-12-31"},
            confidence=0.9
        )

        # Low rigor idea
        low_rigor_idea = IdeaSpec(
            query="Descriptive analysis",
            method="descriptive",
            design_type="descriptive",
            confidence=0.5
        )

        high_scores = scorer.score_idea(high_rigor_idea)
        low_scores = scorer.score_idea(low_rigor_idea)

        assert high_scores["technical_rigor"] > low_scores["technical_rigor"]

    def test_novelty_scoring(self, scorer):
        """Test novelty scoring with previous ideas"""
        idea1 = IdeaSpec(query="First idea", method="event_study")
        idea2 = IdeaSpec(query="Second idea", method="event_study")  # Similar method
        idea3 = IdeaSpec(query="Third idea", method="synthetic_control")  # Different method

        scores1 = scorer.score_idea(idea1)
        scores2 = scorer.score_idea(idea2)
        scores3 = scorer.score_idea(idea3)

        # First idea should have highest novelty
        assert scores1["novelty"] == 1.0

        # Second idea should have lower novelty (similar method)
        assert scores2["novelty"] < scores1["novelty"]

        # Third idea should have higher novelty than second (different method)
        assert scores3["novelty"] > scores2["novelty"]

    def test_execution_result_bonus(self, scorer, sample_idea):
        """Test that successful execution improves technical rigor score"""
        # Score without execution result
        scores_without = scorer.score_idea(sample_idea)

        # Score with successful execution
        execution_result = {
            "success": True,
            "stdout": "Analysis complete with confidence intervals and significance tests"
        }
        scores_with = scorer.score_idea(sample_idea, execution_result)

        assert scores_with["technical_rigor"] > scores_without["technical_rigor"]

    def test_confidence_penalty(self, scorer):
        """Test that low confidence reduces technical rigor score"""
        high_conf_idea = IdeaSpec(query="High confidence analysis", confidence=0.9)
        low_conf_idea = IdeaSpec(query="Low confidence analysis", confidence=0.3)

        high_scores = scorer.score_idea(high_conf_idea)
        low_scores = scorer.score_idea(low_conf_idea)

        # Technical rigor should be affected by confidence
        assert high_scores["technical_rigor"] > low_scores["technical_rigor"]


class TestPortfolioSelector:
    """Test portfolio selection functionality"""

    @pytest.fixture
    def selector(self):
        return PortfolioSelector(diversity_weight=0.3)

    @pytest.fixture
    def sample_ideas(self):
        ideas = []
        for i in range(10):
            idea = IdeaSpec(
                query=f"Analysis idea {i}",
                method=f"method_{i % 3}",  # 3 different methods
                assets=[f"asset_{i % 4}"],  # 4 different assets
                event_context=f"event_{i % 2}" if i % 2 == 0 else None  # 2 different events
            )
            ideas.append(idea)
        return ideas

    @pytest.fixture
    def sample_scores(self, sample_ideas):
        scores = {}
        for i, idea in enumerate(sample_ideas):
            scores[idea.id] = {
                "overall": 0.5 + (i * 0.05),  # Increasing scores
                "domain_relevance": 0.6,
                "trader_value": 0.5,
                "technical_rigor": 0.4,
                "novelty": 0.7
            }
        return scores

    def test_selector_initializes(self, selector):
        """Test that selector initializes properly"""
        assert selector is not None
        assert selector.diversity_weight == 0.3

    def test_select_portfolio_basic(self, selector, sample_ideas, sample_scores):
        """Test basic portfolio selection"""
        portfolio = selector.select_portfolio(sample_ideas, sample_scores, n=3)

        assert isinstance(portfolio, list)
        assert len(portfolio) <= 3
        assert len(portfolio) <= len(sample_ideas)

        # All portfolio items should be from original ideas
        for idea in portfolio:
            assert idea in sample_ideas

    def test_select_portfolio_starts_with_best(self, selector, sample_ideas, sample_scores):
        """Test that portfolio selection starts with highest scoring idea"""
        portfolio = selector.select_portfolio(sample_ideas, sample_scores, n=3)

        if portfolio:
            first_idea = portfolio[0]
            first_score = sample_scores[first_idea.id]["overall"]

            # Should be the highest score (or tied for highest)
            max_score = max(scores["overall"] for scores in sample_scores.values())
            assert first_score == max_score

    def test_select_portfolio_diversity(self, selector, sample_ideas, sample_scores):
        """Test that portfolio selection promotes diversity"""
        portfolio = selector.select_portfolio(sample_ideas, sample_scores, n=5)

        if len(portfolio) > 1:
            # Should have some method diversity
            methods = [idea.method for idea in portfolio]
            assert len(set(methods)) > 1 or len(portfolio) == 1

    def test_select_portfolio_edge_cases(self, selector, sample_scores):
        """Test edge cases in portfolio selection"""
        # Empty ideas list
        portfolio = selector.select_portfolio([], sample_scores, n=3)
        assert portfolio == []

        # Single idea
        single_idea = [IdeaSpec(query="Single idea")]
        single_scores = {single_idea[0].id: {"overall": 0.5}}
        portfolio = selector.select_portfolio(single_idea, single_scores, n=3)
        assert len(portfolio) == 1

        # More ideas requested than available
        few_ideas = [IdeaSpec(query=f"Idea {i}") for i in range(2)]
        few_scores = {idea.id: {"overall": 0.5} for idea in few_ideas}
        portfolio = selector.select_portfolio(few_ideas, few_scores, n=5)
        assert len(portfolio) == 2

    def test_diversity_bonus_calculation(self, selector, sample_ideas):
        """Test diversity bonus calculation"""
        # Test with empty portfolio (should return high bonus)
        bonus = selector._calculate_diversity_bonus(sample_ideas[0], [])
        assert bonus == 1.0

        # Test with existing portfolio
        portfolio = [sample_ideas[0]]
        bonus = selector._calculate_diversity_bonus(sample_ideas[1], portfolio)
        assert isinstance(bonus, (int, float))
        assert 0.0 <= bonus <= 1.2  # Maximum possible bonus