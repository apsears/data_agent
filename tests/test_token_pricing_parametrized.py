#!/usr/bin/env python3
"""
Parametrized test for token pricing calculation using pricing data files.
Tests all possible scenarios that could occur in the application.
"""
import sys
from pathlib import Path
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from native_transparent_agent import calculate_token_cost


class TestTokenPricingParametrized:
    """Test token pricing with parametrized scenarios"""

    @pytest.mark.parametrize("model_name,input_tokens,output_tokens,expected_cost,description", [
        # GPT-5 Mini scenarios (as found in config and logs)
        ("gpt-5-mini", 1_000_000, 1_000_000, 2.25, "GPT-5 Mini without prefix (as in config)"),
        ("openai:gpt-5-mini", 1_000_000, 1_000_000, 2.25, "GPT-5 Mini with openai: prefix"),
        ("gpt-5-mini", 100_000, 50_000, 0.125, "GPT-5 Mini smaller token count"),
        ("gpt-5-mini", 1_000, 2_000, 0.00425, "GPT-5 Mini tiny token count"),

        # Claude 4 Sonnet scenarios
        ("claude-sonnet-4-20250514", 1_000_000, 1_000_000, 18.00, "Claude 4 Sonnet without prefix"),
        ("anthropic:claude-sonnet-4-20250514", 1_000_000, 1_000_000, 18.00, "Claude 4 Sonnet with prefix"),
        ("claude-sonnet-4-20250514", 10_000, 5_000, 0.105, "Claude 4 Sonnet smaller count"),

        # Other OpenAI models
        ("gpt-5", 1_000_000, 1_000_000, 11.25, "GPT-5 standard"),
        ("openai:gpt-5", 1_000_000, 1_000_000, 11.25, "GPT-5 with prefix"),
        ("gpt-4o", 1_000_000, 1_000_000, 12.50, "GPT-4o"),
        ("gpt-4o-mini", 1_000_000, 1_000_000, 0.75, "GPT-4o-mini"),

        # Other Anthropic models
        ("claude-3-5-haiku-20241022", 1_000_000, 1_000_000, 4.80, "Claude 3.5 Haiku"),
        ("anthropic:claude-3-5-haiku-20241022", 1_000_000, 1_000_000, 4.80, "Claude 3.5 Haiku with prefix"),

        # Edge cases with zero tokens
        ("gpt-5-mini", 0, 0, 0.0, "Zero tokens"),
        ("gpt-5-mini", 1000, 0, 0.00025, "Only input tokens"),
        ("gpt-5-mini", 0, 1000, 0.002, "Only output tokens"),
    ])
    def test_pricing_scenarios(self, model_name, input_tokens, output_tokens, expected_cost, description):
        """Test various pricing scenarios with different models and token counts"""
        cost = calculate_token_cost(model_name, input_tokens, output_tokens)
        assert abs(cost - expected_cost) < 0.00001, f"Failed for {description}: expected {expected_cost}, got {cost}"

    @pytest.mark.parametrize("invalid_model", [
        "nonexistent-model",
        "gpt-99",
        "claude-unknown",
        "random-text"
    ])
    def test_invalid_models(self, invalid_model):
        """Test that invalid models raise appropriate errors"""
        with pytest.raises(ValueError, match="Pricing data not found"):
            calculate_token_cost(invalid_model, 1000, 1000)

    @pytest.mark.parametrize("model_base,prefixes", [
        ("gpt-5-mini", ["", "openai:"]),
        ("claude-sonnet-4-20250514", ["", "anthropic:"]),
        ("gpt-5", ["", "openai:"]),
        ("claude-3-5-haiku-20241022", ["", "anthropic:"])
    ])
    def test_prefix_consistency(self, model_base, prefixes):
        """Test that models work consistently with and without prefixes"""
        costs = []
        for prefix in prefixes:
            model_name = f"{prefix}{model_base}"
            cost = calculate_token_cost(model_name, 10000, 10000)
            costs.append(cost)

        # All costs should be identical regardless of prefix
        assert all(abs(c - costs[0]) < 0.00001 for c in costs), f"Inconsistent pricing for {model_base} with different prefixes"

    def test_critic_model_from_config(self):
        """Test the exact scenario that failed in production: critic_model from config"""
        # This is exactly what happens when the config specifies critic_model: "gpt-5-mini"
        critic_model = "gpt-5-mini"  # As specified in config/config.yaml

        # Simulating the critic's token usage calculation
        input_tokens = 5000
        output_tokens = 2000

        # This should work without raising an error
        cost = calculate_token_cost(critic_model, input_tokens, output_tokens)

        # Verify the cost is correct: (5000/1M * 0.25) + (2000/1M * 2.00)
        expected_cost = (5000 / 1_000_000 * 0.25) + (2000 / 1_000_000 * 2.00)
        assert abs(cost - expected_cost) < 0.00001, f"Critic model pricing failed: expected {expected_cost}, got {cost}"

    def test_real_world_token_counts(self):
        """Test with realistic token counts from actual runs"""
        test_cases = [
            # From the actual ReAct log
            ("claude-sonnet-4-20250514", 6200, 107, 0.020204999999999997),
            ("claude-sonnet-4-20250514", 6473, 2317, 0.054174),
            ("claude-sonnet-4-20250514", 18702, 2573, 0.09470100000000001),
        ]

        for model, input_tokens, output_tokens, expected_cost in test_cases:
            cost = calculate_token_cost(model, input_tokens, output_tokens)
            # Allow small floating point differences
            assert abs(cost - expected_cost) < 0.0001, f"Real-world test failed for {model}: expected {expected_cost}, got {cost}"


if __name__ == "__main__":
    # Run pytest with verbose output
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"])