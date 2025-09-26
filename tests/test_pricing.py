"""
Tests for pricing and token counting functionality.
"""

import pytest
import tempfile
from pathlib import Path
import pandas as pd
from unittest.mock import patch, mock_open

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from run_batch_queries import load_pricing_data, count_tokens


class TestPricingData:
    """Test cases for pricing data loading."""

    def test_load_anthropic_pricing_success(self):
        """Test successful loading of Anthropic pricing data from real file."""
        # Test Anthropic model using real pricing file
        pricing = load_pricing_data("anthropic:claude-sonnet-4-20250514")

        # Verify we get expected models from the real file
        assert len(pricing) > 0
        assert "claude-haiku-3-5-20241022" in pricing
        assert pricing["claude-haiku-3-5-20241022"]["input_per_1m"] == 0.80
        assert pricing["claude-haiku-3-5-20241022"]["output_per_1m"] == 4.00

    def test_load_openai_pricing_success(self):
        """Test successful loading of OpenAI pricing data from real file."""
        # Test OpenAI model using real pricing file
        pricing = load_pricing_data("openai:gpt-4o-mini")

        # Verify we get expected models from the real file
        assert len(pricing) > 0
        assert "gpt-4o-mini" in pricing
        assert pricing["gpt-4o-mini"]["input_per_1m"] == 0.15
        assert pricing["gpt-4o-mini"]["output_per_1m"] == 0.60

    def test_load_pricing_file_not_found(self):
        """Test error handling when pricing file doesn't exist."""
        with patch('run_batch_queries.Path') as mock_path:
            mock_path.return_value.exists.return_value = False

            with pytest.raises(FileNotFoundError, match="Pricing file not found"):
                load_pricing_data("anthropic:claude-sonnet-4")

    def test_load_pricing_empty_file(self):
        """Test error handling when pricing file is empty."""
        empty_tsv_content = """Model	Input	Output"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write(empty_tsv_content)
            temp_file = f.name

        with patch('run_batch_queries.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch('run_batch_queries.pd.read_csv') as mock_read_csv:
                mock_df = pd.read_csv(temp_file, sep='\t')
                mock_read_csv.return_value = mock_df

                with pytest.raises(ValueError, match="No valid pricing data found"):
                    load_pricing_data("anthropic:claude-sonnet-4")

        # Clean up
        Path(temp_file).unlink()

    def test_model_provider_detection(self):
        """Test that correct pricing file is selected based on model provider."""
        anthropic_tsv_content = """Model	Input	Output
claude-haiku-3.5	0.80	4.00"""

        openai_tsv_content = """Model	Input	Output
gpt-4o-mini	$0.15	$0.60"""

        # Test Anthropic model selection
        with patch('run_batch_queries.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch('run_batch_queries.pd.read_csv') as mock_read_csv:
                mock_df = pd.DataFrame({'Model': ['claude-haiku-3.5'], 'Input': ['0.80'], 'Output': ['4.00']})
                mock_read_csv.return_value = mock_df

                pricing = load_pricing_data("anthropic:claude-haiku-3.5")
                assert "claude-haiku-3.5" in pricing

        # Test OpenAI model selection
        with patch('run_batch_queries.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch('run_batch_queries.pd.read_csv') as mock_read_csv:
                mock_df = pd.DataFrame({'Model': ['gpt-4o-mini'], 'Input': ['$0.15'], 'Output': ['$0.60']})
                mock_read_csv.return_value = mock_df

                pricing = load_pricing_data("openai:gpt-4o-mini")
                assert "gpt-4o-mini" in pricing


class TestTokenCounting:
    """Test cases for token counting functionality."""

    def test_count_tokens_basic(self):
        """Test basic token counting functionality."""
        text = "Hello world"
        tokens = count_tokens(text, "gpt-4o-mini")

        # Should return a positive integer
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_count_tokens_empty_string(self):
        """Test token counting with empty string."""
        tokens = count_tokens("", "gpt-4o-mini")
        assert tokens == 0

    def test_count_tokens_different_models(self):
        """Test token counting with different model encodings."""
        text = "Hello world, this is a test message."

        # Test different models
        tokens_gpt4o = count_tokens(text, "gpt-4o-mini")
        tokens_gpt4 = count_tokens(text, "gpt-4")
        tokens_gpt35 = count_tokens(text, "gpt-3.5-turbo")

        # All should return positive integers
        assert all(isinstance(t, int) and t > 0 for t in [tokens_gpt4o, tokens_gpt4, tokens_gpt35])

    def test_count_tokens_fallback(self):
        """Test token counting fallback for unknown models."""
        text = "Hello world"
        tokens = count_tokens(text, "unknown-model")

        # Should fall back to character-based estimation
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_count_tokens_long_text(self):
        """Test token counting with longer text."""
        long_text = "This is a longer text " * 100  # 400+ words
        tokens = count_tokens(long_text, "gpt-4o-mini")

        # Should handle long text appropriately
        assert isinstance(tokens, int)
        assert tokens > 100  # Should be significantly more than short text

    def test_count_tokens_special_characters(self):
        """Test token counting with special characters and unicode."""
        special_text = "Hello 世界! 🚀 $100 @user #hashtag"
        tokens = count_tokens(special_text, "gpt-4o-mini")

        assert isinstance(tokens, int)
        assert tokens > 0


class TestPricingIntegration:
    """Integration tests for pricing and cost calculation."""

    def test_cost_calculation_anthropic(self):
        """Test accurate cost calculation for Anthropic models."""
        anthropic_tsv_content = """Model	Input	Output
claude-haiku-3-5-20241022	0.80	4.00"""

        with patch('run_batch_queries.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch('run_batch_queries.pd.read_csv') as mock_read_csv:
                mock_df = pd.DataFrame({
                    'Model': ['claude-haiku-3-5-20241022'],
                    'Input': ['0.80'],
                    'Output': ['4.00']
                })
                mock_read_csv.return_value = mock_df

                pricing = load_pricing_data("anthropic:claude-haiku-3-5-20241022")
                model_pricing = pricing["claude-haiku-3-5-20241022"]

                # Test cost calculation
                input_tokens = 1000
                output_tokens = 500

                input_cost = input_tokens * model_pricing["input_per_1m"] / 1_000_000
                output_cost = output_tokens * model_pricing["output_per_1m"] / 1_000_000
                total_cost = input_cost + output_cost

                expected_total = (1000 * 0.80 + 500 * 4.00) / 1_000_000
                assert abs(total_cost - expected_total) < 1e-10

    def test_missing_model_error(self):
        """Test error when requesting pricing for non-existent model."""
        anthropic_tsv_content = """Model	Input	Output
claude-haiku-3.5	0.80	4.00"""

        with patch('run_batch_queries.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch('run_batch_queries.pd.read_csv') as mock_read_csv:
                mock_df = pd.DataFrame({
                    'Model': ['claude-haiku-3.5'],
                    'Input': ['0.80'],
                    'Output': ['4.00']
                })
                mock_read_csv.return_value = mock_df

                pricing = load_pricing_data("anthropic:claude-haiku-3.5")

                # Should have the model we loaded
                assert "claude-haiku-3.5" in pricing

                # Should raise error for missing model
                with pytest.raises(KeyError):
                    pricing["non-existent-model"]


if __name__ == "__main__":
    pytest.main([__file__])