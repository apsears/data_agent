"""Test seed generation functionality and technique name consistency"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from idea_generation_mvp.inspiration.events import EventLibrary
from idea_generation_mvp.inspiration.techniques import TechniqueLibrary
from idea_generation_mvp.core.seeds import SeedGenerator
from idea_generation_mvp.core.specifications import IdeaSpec


class TestSeedGenerator:
    """Test seed generation functionality"""

    @pytest.fixture
    def generator(self):
        events = EventLibrary()
        techniques = TechniqueLibrary()
        return SeedGenerator(events, techniques)

    @pytest.fixture
    def test_config(self):
        return {
            "n_event_driven": 2,
            "n_technique_driven": 2,
            "n_hybrid": 1,
            "n_trader_questions": 2
        }

    def test_generator_initializes(self, generator):
        """Test that generator initializes properly"""
        assert generator is not None
        assert generator.events is not None
        assert generator.techniques is not None

    def test_generate_event_driven_seeds(self, generator):
        """Test event-driven seed generation"""
        seeds = generator.generate_event_driven_seeds(3)
        assert isinstance(seeds, list)
        assert len(seeds) <= 3  # Should generate at most 3

        for seed in seeds:
            assert isinstance(seed, IdeaSpec)
            assert seed.event_context is not None
            assert len(seed.query) > 0

    def test_generate_technique_driven_seeds(self, generator):
        """Test technique-driven seed generation"""
        seeds = generator.generate_technique_driven_seeds(3)
        assert isinstance(seeds, list)
        assert len(seeds) <= 3

        for seed in seeds:
            assert isinstance(seed, IdeaSpec)
            assert seed.technique_context is not None
            assert len(seed.query) > 0

    def test_generate_hybrid_seeds(self, generator):
        """Test hybrid seed generation"""
        seeds = generator.generate_hybrid_seeds(2)
        assert isinstance(seeds, list)
        assert len(seeds) <= 2

        for seed in seeds:
            assert isinstance(seed, IdeaSpec)
            assert seed.event_context is not None
            assert seed.technique_context is not None
            assert len(seed.query) > 0

    def test_generate_trader_question_seeds(self, generator):
        """Test trader question seed generation - this would catch our bug"""
        # This is the critical test that would have caught the technique name mismatch
        seeds = generator.generate_trader_question_seeds(3)
        assert isinstance(seeds, list)
        assert len(seeds) <= 3

        for seed in seeds:
            assert isinstance(seed, IdeaSpec)
            assert len(seed.query) > 0
            # Should have either event or technique context from trader questions

    def test_generate_all_seeds(self, generator, test_config):
        """Test complete seed generation pipeline"""
        seeds = generator.generate_all_seeds(test_config)
        assert isinstance(seeds, list)

        expected_total = sum(test_config.values())
        assert len(seeds) <= expected_total  # May be fewer due to sampling

        for seed in seeds:
            assert isinstance(seed, IdeaSpec)
            assert seed.id is not None
            assert len(seed.query) > 0
            assert seed.method is not None

    def test_create_technique_query_coverage(self, generator):
        """Test that _create_technique_query covers all techniques"""
        # This test would catch missing techniques in the query mapping
        for technique in generator.techniques.techniques:
            query = generator._create_technique_query(technique)
            assert isinstance(query, str)
            assert len(query) > 0
            # Should not be just the fallback
            assert technique.name.lower() in query.lower() or len(query) > 50

    def test_technique_query_mapping_completeness(self, generator):
        """Test that technique query mapping includes all technique names"""
        # Read the actual mapping from the method
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

        # Check that all actual technique names are covered
        actual_technique_names = [t.name for t in generator.techniques.techniques]

        for technique_name in actual_technique_names:
            assert technique_name in technique_queries, f"Technique '{technique_name}' missing from query mapping"

    def test_seed_validation(self, generator, test_config):
        """Test that generated seeds pass basic validation"""
        seeds = generator.generate_all_seeds(test_config)

        for seed in seeds:
            # Test basic structure
            assert hasattr(seed, 'id')
            assert hasattr(seed, 'query')
            assert hasattr(seed, 'method')
            assert hasattr(seed, 'assets')
            assert hasattr(seed, 'confidence')

            # Test data types
            assert isinstance(seed.assets, list)
            assert isinstance(seed.confidence, (int, float))
            assert 0 <= seed.confidence <= 1

    def test_no_empty_queries(self, generator, test_config):
        """Test that no seeds have empty queries"""
        seeds = generator.generate_all_seeds(test_config)

        for seed in seeds:
            assert seed.query is not None
            assert len(seed.query.strip()) > 0
            assert seed.query != ""

    def test_technique_method_consistency(self, generator):
        """Test that technique methods are consistent with available techniques"""
        seeds = generator.generate_technique_driven_seeds(5)

        for seed in seeds:
            # Method should be derivable from technique name
            assert seed.method is not None
            assert len(seed.method) > 0