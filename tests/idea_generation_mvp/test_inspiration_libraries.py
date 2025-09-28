"""Test inspiration libraries for consistency and completeness"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from idea_generation_mvp.inspiration.events import EventLibrary, Event
from idea_generation_mvp.inspiration.techniques import TechniqueLibrary, AnalysisTechnique


class TestEventLibrary:
    """Test event library functionality"""

    @pytest.fixture
    def event_library(self):
        return EventLibrary()

    def test_event_library_loads(self, event_library):
        """Test that event library loads without errors"""
        assert event_library is not None
        assert len(event_library.events) > 0

    def test_events_have_required_fields(self, event_library):
        """Test that all events have required fields"""
        for event in event_library.events:
            assert isinstance(event, Event)
            assert event.name is not None and len(event.name) > 0
            assert event.date is not None and len(event.date) > 0
            assert event.description is not None and len(event.description) > 0
            assert event.impact_type is not None and len(event.impact_type) > 0
            assert event.affected_assets is not None and len(event.affected_assets) > 0
            assert event.trader_value is not None and len(event.trader_value) > 0

    def test_event_dates_valid_format(self, event_library):
        """Test that event dates are in reasonable format"""
        for event in event_library.events:
            # Check that date contains year 20XX
            assert "20" in event.date or "2019" in event.date or "2021" in event.date or "2022" in event.date or "2023" in event.date or "2024" in event.date

    def test_minimum_event_count(self, event_library):
        """Test that we have minimum expected number of events"""
        assert len(event_library.events) >= 15  # Should have at least 15 major events

    def test_get_events_by_type(self, event_library):
        """Test event filtering by type"""
        outage_events = event_library.get_events_by_type("outage")
        assert isinstance(outage_events, list)

        weather_events = event_library.get_events_by_type("weather")
        assert isinstance(weather_events, list)

    def test_no_duplicate_event_names(self, event_library):
        """Test that event names are unique"""
        event_names = [event.name for event in event_library.events]
        assert len(event_names) == len(set(event_names))


class TestTechniqueLibrary:
    """Test technique library functionality"""

    @pytest.fixture
    def technique_library(self):
        return TechniqueLibrary()

    def test_technique_library_loads(self, technique_library):
        """Test that technique library loads without errors"""
        assert technique_library is not None
        assert len(technique_library.techniques) > 0

    def test_techniques_have_required_fields(self, technique_library):
        """Test that all techniques have required fields"""
        for technique in technique_library.techniques:
            assert isinstance(technique, AnalysisTechnique)
            assert technique.name is not None and len(technique.name) > 0
            assert technique.method_type is not None and len(technique.method_type) > 0
            assert technique.description is not None and len(technique.description) > 0
            assert technique.trader_value is not None and len(technique.trader_value) > 0
            assert technique.implementation is not None and len(technique.implementation) > 0
            assert isinstance(technique.required_data, list)
            assert isinstance(technique.output_artifacts, list)
            assert isinstance(technique.robustness_checks, list)

    def test_minimum_technique_count(self, technique_library):
        """Test that we have minimum expected number of techniques"""
        assert len(technique_library.techniques) >= 15  # Should have at least 15 techniques

    def test_get_technique_by_type(self, technique_library):
        """Test technique filtering by type"""
        causal_techniques = technique_library.get_technique_by_type("causal")
        assert isinstance(causal_techniques, list)
        assert len(causal_techniques) > 0

        pattern_techniques = technique_library.get_technique_by_type("pattern")
        assert isinstance(pattern_techniques, list)

    def test_no_duplicate_technique_names(self, technique_library):
        """Test that technique names are unique"""
        technique_names = [technique.name for technique in technique_library.techniques]
        assert len(technique_names) == len(set(technique_names))

    def test_trader_questions_exist(self, technique_library):
        """Test that trader questions are properly loaded"""
        assert hasattr(technique_library, 'trader_questions')
        assert isinstance(technique_library.trader_questions, list)
        assert len(technique_library.trader_questions) > 0

        for question in technique_library.trader_questions:
            assert "question" in question
            assert "method" in question
            assert "value" in question

    def test_suggest_technique_for_question_no_errors(self, technique_library):
        """Test that suggest_technique_for_question works for all trader questions"""
        for trader_q in technique_library.trader_questions:
            # This is the critical test that would have caught our bug
            result = technique_library.suggest_technique_for_question(trader_q["question"])
            assert result is not None
            assert isinstance(result, AnalysisTechnique)
            assert result.name is not None

    def test_suggest_technique_constraint_keywords(self, technique_library):
        """Test constraint keyword matching specifically"""
        # Test various constraint-related questions
        constraint_questions = [
            "Where do constraints bind?",
            "What capacity limits exist?",
            "How do constrained flows behave?",
            "Which locations show capacity constraints?"
        ]

        for question in constraint_questions:
            result = technique_library.suggest_technique_for_question(question)
            assert result is not None
            # Should find the Utilization Constraint Detection technique
            assert "constraint" in result.name.lower() or "constraint" in result.description.lower()

    def test_technique_name_consistency(self, technique_library):
        """Test that all technique names are consistent with what's expected"""
        technique_names = [t.name for t in technique_library.techniques]

        # Verify some key expected techniques exist
        expected_techniques = [
            "Event Study DiD",
            "Synthetic Control",
            "Utilization Constraint Detection",
            "Panel VAR Lead-Lag"
        ]

        for expected in expected_techniques:
            assert any(expected in name for name in technique_names), f"Expected technique '{expected}' not found"

    def test_method_types_valid(self, technique_library):
        """Test that method types are from expected categories"""
        valid_method_types = ["causal", "pattern", "anomaly", "forecasting", "optimization"]

        for technique in technique_library.techniques:
            assert technique.method_type in valid_method_types, f"Invalid method type: {technique.method_type}"