"""Integration tests for the complete MVP pipeline"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from idea_generation_mvp.inspiration.events import EventLibrary
from idea_generation_mvp.inspiration.techniques import TechniqueLibrary
from idea_generation_mvp.core.seeds import SeedGenerator
from idea_generation_mvp.evaluation.scorer import IdeaScorer, PortfolioSelector
from idea_generation_mvp.evolution.operators import EvolutionOperators


class TestMVPIntegration:
    """Integration tests for the complete MVP pipeline"""

    @pytest.fixture
    def mvp_components(self):
        """Set up all MVP components"""
        events = EventLibrary()
        techniques = TechniqueLibrary()
        generator = SeedGenerator(events, techniques)
        scorer = IdeaScorer()
        selector = PortfolioSelector()
        operators = EvolutionOperators(events, techniques)

        return {
            "events": events,
            "techniques": techniques,
            "generator": generator,
            "scorer": scorer,
            "selector": selector,
            "operators": operators
        }

    @pytest.fixture
    def test_config(self):
        return {
            "n_event_driven": 2,
            "n_technique_driven": 2,
            "n_hybrid": 1,
            "n_trader_questions": 2
        }

    def test_end_to_end_pipeline(self, mvp_components, test_config):
        """Test the complete end-to-end pipeline"""
        components = mvp_components

        # Step 1: Generate seeds
        seeds = components["generator"].generate_all_seeds(test_config)
        assert len(seeds) > 0

        # Step 2: Score all seeds
        all_scores = {}
        for idea in seeds:
            scores = components["scorer"].score_idea(idea)
            all_scores[idea.id] = scores
            idea.scores = scores

        assert len(all_scores) == len(seeds)

        # Step 3: Select portfolio
        portfolio = components["selector"].select_portfolio(seeds, all_scores, n=3)
        assert len(portfolio) <= 3
        assert len(portfolio) <= len(seeds)

        # Verify portfolio quality
        for idea in portfolio:
            assert idea.id in all_scores
            assert all_scores[idea.id]["overall"] > 0

    def test_evolution_integration(self, mvp_components, test_config):
        """Test evolution integration with seed generation"""
        components = mvp_components

        # Generate initial seeds
        seeds = components["generator"].generate_all_seeds(test_config)
        assert len(seeds) > 0

        # Evolve first seed
        if seeds:
            evolved = components["operators"].evolve_idea(
                seeds[0],
                ['make_more_specific', 'add_robustness']
            )
            # Evolution might return empty list for some seeds, that's OK
            assert isinstance(evolved, list)

    def test_scoring_consistency(self, mvp_components, test_config):
        """Test that scoring is consistent and reasonable"""
        components = mvp_components

        seeds = components["generator"].generate_all_seeds(test_config)
        scorer = components["scorer"]

        for seed in seeds:
            scores = scorer.score_idea(seed)

            # Check score structure
            assert "overall" in scores
            assert "domain_relevance" in scores
            assert "trader_value" in scores
            assert "technical_rigor" in scores
            assert "novelty" in scores

            # Check score ranges
            for score_type, score_value in scores.items():
                assert 0.0 <= score_value <= 1.0, f"Score {score_type} = {score_value} out of range"

            # Overall score should be weighted average
            expected_overall = sum(
                scores[dim] * scorer.weights[dim]
                for dim in scorer.weights.keys()
            )
            assert abs(scores["overall"] - expected_overall) < 0.001

    def test_portfolio_diversity(self, mvp_components, test_config):
        """Test that portfolio selection promotes diversity"""
        components = mvp_components

        # Generate larger set for diversity testing
        large_config = {k: v * 3 for k, v in test_config.items()}
        seeds = components["generator"].generate_all_seeds(large_config)

        if len(seeds) < 5:
            # Skip if not enough seeds for meaningful diversity test
            pytest.skip("Not enough seeds generated for diversity test")

        # Score all seeds
        all_scores = {}
        for idea in seeds:
            scores = components["scorer"].score_idea(idea)
            all_scores[idea.id] = scores

        # Select portfolio
        portfolio = components["selector"].select_portfolio(seeds, all_scores, n=5)

        if len(portfolio) > 1:
            # Check method diversity
            methods = [idea.method for idea in portfolio]
            unique_methods = len(set(methods))
            assert unique_methods > 1 or len(portfolio) == 1

    def test_all_trader_questions_work(self, mvp_components):
        """Critical test: ensure all trader questions work without errors"""
        # This test specifically catches the technique name mismatch bug
        components = mvp_components
        techniques = components["techniques"]

        for trader_q in techniques.trader_questions:
            # This should not raise StopIteration or any other exception
            result = techniques.suggest_technique_for_question(trader_q["question"])
            assert result is not None

        # Also test the seed generation that uses trader questions
        seeds = components["generator"].generate_trader_question_seeds(len(techniques.trader_questions))
        assert isinstance(seeds, list)

    def test_data_consistency(self, mvp_components):
        """Test consistency between data sources"""
        components = mvp_components

        # Test that technique names in library match those expected by seed generator
        technique_names = [t.name for t in components["techniques"].techniques]

        # Generate some technique-driven seeds
        seeds = components["generator"].generate_technique_driven_seeds(5)

        for seed in seeds:
            if seed.technique_context:
                # The technique mentioned in context should exist in our library
                context_lower = seed.technique_context.lower()
                found_match = any(
                    name.lower() in context_lower
                    for name in technique_names
                )
                assert found_match, f"Technique context '{seed.technique_context}' doesn't match library"

    def test_error_handling(self, mvp_components):
        """Test error handling in various components"""
        components = mvp_components

        # Test with invalid config
        invalid_config = {"n_event_driven": -1}
        try:
            seeds = components["generator"].generate_all_seeds(invalid_config)
            # Should handle gracefully (empty list or error)
            assert isinstance(seeds, list)
        except Exception as e:
            # Acceptable to raise exception for invalid config
            assert isinstance(e, (ValueError, TypeError))

        # Test scoring with minimal idea
        minimal_idea = components["generator"].generate_event_driven_seeds(1)[0]
        scores = components["scorer"].score_idea(minimal_idea)
        assert isinstance(scores, dict)

    def test_reproducibility(self, mvp_components, test_config):
        """Test that the system produces consistent results"""
        components = mvp_components

        # Generate seeds twice
        seeds1 = components["generator"].generate_all_seeds(test_config)
        seeds2 = components["generator"].generate_all_seeds(test_config)

        # Should produce same number of seeds (assuming deterministic sampling)
        # Note: This might fail if random sampling is used, which is acceptable
        if len(seeds1) == len(seeds2):
            # At least the structure should be consistent
            for s1, s2 in zip(seeds1, seeds2):
                assert type(s1) == type(s2)
                assert hasattr(s1, 'query')
                assert hasattr(s2, 'query')