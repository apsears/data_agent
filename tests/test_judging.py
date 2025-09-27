"""
Tests for LLM judging functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from run_batch_queries import judge_single_result


class TestJudging:
    """Test cases for LLM judging functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_result = {
            "query_id": "test_001",
            "query": "Test query",
            "category": "test",
            "success": True,
            "execution_time": 10.0,
            "final_answer": json.dumps({
                "answer": "The answer is 42",
                "artifacts": ["test.py"],
                "confidence": 0.95
            }),
            "run_directory": ".runs/test",
            "return_code": 0
        }

        self.sample_query_data = {
            "id": "test_001",
            "query": "Test query",
            "category": "test",
            "expected_answer": "42"
        }

        self.sample_config = {
            "judging": {
                "model": "anthropic:claude-haiku-3-5-20241022",
                "template": "templates/judging_prompt.txt",
                "timeout_sec": 30
            }
        }

    def test_judge_single_result_failed_query(self):
        """Test judging behavior when the original query failed."""
        failed_result = self.sample_result.copy()
        failed_result["success"] = False

        result = judge_single_result(failed_result, self.sample_query_data, self.sample_config)

        assert result["judging_performed"] is False
        assert result["accuracy_score"] == 0
        assert "Query execution failed" in result["explanation"]
        assert result["judging_cost"]["total_cost"] == 0.0

    def test_judge_single_result_missing_template(self):
        """Test judging behavior when template file is missing."""
        with patch('run_batch_queries.Path') as mock_path:
            mock_path.return_value.exists.return_value = False

            result = judge_single_result(self.sample_result, self.sample_query_data, self.sample_config)

            assert result["judging_performed"] is False
            assert "template not found" in result["error"]
            assert result["judging_cost"]["total_cost"] == 0.0

    @patch('run_batch_queries.subprocess.run')
    @patch('run_batch_queries.Path')
    @patch('run_batch_queries.count_tokens')
    @patch('run_batch_queries.load_pricing_data')
    def test_judge_single_result_success(self, mock_load_pricing, mock_count_tokens, mock_path, mock_subprocess):
        """Test successful judging execution."""
        # Mock template file exists
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.read_text.return_value = "Mock template content with {{query}} and {{actual_answer}}"

        # Mock token counting
        mock_count_tokens.side_effect = [100, 50]  # input_tokens, output_tokens

        # Mock pricing data
        mock_load_pricing.return_value = {
            "claude-haiku-3-5-20241022": {
                "input_per_1m": 0.80,
                "output_per_1m": 4.00
            }
        }

        # Mock successful subprocess execution
        mock_subprocess_result = MagicMock()
        mock_subprocess_result.returncode = 0
        mock_subprocess_result.stdout = '''
Starting ReAct task...
=== FINAL ANSWER ===
{
  "answer": "{\\"accuracy_score\\": 5, \\"explanation\\": \\"Perfect match\\", \\"confidence\\": 1.0}"
}
'''
        mock_subprocess.return_value = mock_subprocess_result

        result = judge_single_result(self.sample_result, self.sample_query_data, self.sample_config)

        assert result["judging_performed"] is True
        assert result["accuracy_score"] == 5
        assert result["explanation"] == "Perfect match"
        assert result["confidence"] == 1.0

        # Check cost calculation
        expected_cost = (100 * 0.80 + 50 * 4.00) / 1_000_000
        assert abs(result["judging_cost"]["total_cost"] - expected_cost) < 1e-10
        assert result["judging_cost"]["input_tokens"] == 100
        assert result["judging_cost"]["output_tokens"] == 50

    @patch('run_batch_queries.subprocess.run')
    @patch('run_batch_queries.Path')
    @patch('run_batch_queries.count_tokens')
    @patch('run_batch_queries.load_pricing_data')
    def test_judge_single_result_subprocess_failure(self, mock_load_pricing, mock_count_tokens, mock_path, mock_subprocess):
        """Test judging behavior when subprocess execution fails."""
        # Mock template file exists
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.read_text.return_value = "Mock template"

        # Mock token counting
        mock_count_tokens.return_value = 100

        # Mock pricing data
        mock_load_pricing.return_value = {
            "claude-haiku-3-5-20241022": {
                "input_per_1m": 0.80,
                "output_per_1m": 4.00
            }
        }

        # Mock failed subprocess execution
        mock_subprocess_result = MagicMock()
        mock_subprocess_result.returncode = 1
        mock_subprocess_result.stderr = "Model not found error"
        mock_subprocess.return_value = mock_subprocess_result

        result = judge_single_result(self.sample_result, self.sample_query_data, self.sample_config)

        assert result["judging_performed"] is False
        assert "Judging execution failed" in result["error"]
        assert "Model not found error" in result["error"]

    @patch('run_batch_queries.subprocess.run')
    @patch('run_batch_queries.Path')
    @patch('run_batch_queries.count_tokens')
    @patch('run_batch_queries.load_pricing_data')
    def test_judge_single_result_timeout(self, mock_load_pricing, mock_count_tokens, mock_path, mock_subprocess):
        """Test judging behavior when subprocess times out."""
        # Mock template file exists
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.read_text.return_value = "Mock template"

        # Mock token counting
        mock_count_tokens.return_value = 100

        # Mock pricing data
        mock_load_pricing.return_value = {
            "claude-haiku-3-5-20241022": {
                "input_per_1m": 0.80,
                "output_per_1m": 4.00
            }
        }

        # Mock subprocess timeout
        from subprocess import TimeoutExpired
        mock_subprocess.side_effect = TimeoutExpired("cmd", 60)

        result = judge_single_result(self.sample_result, self.sample_query_data, self.sample_config)

        assert result["judging_performed"] is False
        assert "timed out" in result["error"]

    def test_answer_extraction_json_format(self):
        """Test extraction of answer from JSON formatted final_answer."""
        result_with_json = self.sample_result.copy()
        result_with_json["final_answer"] = json.dumps({
            "answer": "The extracted answer",
            "confidence": 0.9
        })

        with patch('run_batch_queries.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.read_text.return_value = "Template with {{actual_answer}}"

            with patch('run_batch_queries.count_tokens') as mock_count_tokens:
                mock_count_tokens.return_value = 50

                with patch('run_batch_queries.load_pricing_data') as mock_pricing:
                    mock_pricing.return_value = {
                        "claude-haiku-3-5-20241022": {"input_per_1m": 0.80, "output_per_1m": 4.00}
                    }

                    with patch('run_batch_queries.subprocess.run') as mock_subprocess:
                        # The judging function should extract "The extracted answer" from the JSON
                        # We can verify this by checking what gets passed to the template
                        mock_subprocess_result = MagicMock()
                        mock_subprocess_result.returncode = 0
                        mock_subprocess_result.stdout = '=== FINAL ANSWER ===\n{"answer": "{}"}'
                        mock_subprocess.return_value = mock_subprocess_result

                        judge_single_result(result_with_json, self.sample_query_data, self.sample_config)

                        # Verify subprocess was called (indicating successful template rendering)
                        assert mock_subprocess.called

    @patch('run_batch_queries.load_pricing_data')
    def test_missing_model_pricing(self, mock_load_pricing):
        """Test error handling when model pricing is not found."""
        # Mock pricing data without our model
        mock_load_pricing.return_value = {
            "different-model": {"input_per_1m": 1.0, "output_per_1m": 5.0}
        }

        with patch('run_batch_queries.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.read_text.return_value = "Template"

            with patch('run_batch_queries.count_tokens') as mock_count_tokens:
                mock_count_tokens.return_value = 100

                with patch('run_batch_queries.subprocess.run') as mock_subprocess:
                    mock_subprocess_result = MagicMock()
                    mock_subprocess_result.returncode = 0
                    mock_subprocess_result.stdout = '=== FINAL ANSWER ===\n{"answer": "{}"}'
                    mock_subprocess.return_value = mock_subprocess_result

                    result = judge_single_result(self.sample_result, self.sample_query_data, self.sample_config)

                    assert result["judging_performed"] is False
                    assert "Pricing not found for model" in result["error"]


class TestJudgingIntegration:
    """Integration tests for judging with real file operations."""

    def test_template_rendering(self):
        """Test that Jinja2 template rendering works correctly."""
        # Create a temporary template file
        template_content = """Query: {{ query }}
Expected: {{ expected_answer }}
Actual: {{ actual_answer }}"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(template_content)
            template_file = f.name

        try:
            config = {
                "judging": {
                    "model": "anthropic:claude-haiku-3-5-20241022",
                    "template": template_file,
                    "timeout_sec": 30
                }
            }

            result = {
                "success": True,
                "final_answer": "Simple answer"
            }

            query_data = {
                "query": "Test query",
                "expected_answer": "Expected result"
            }

            with patch('run_batch_queries.count_tokens') as mock_count_tokens:
                mock_count_tokens.return_value = 50

                with patch('run_batch_queries.load_pricing_data') as mock_pricing:
                    mock_pricing.return_value = {
                        "claude-haiku-3-5-20241022": {"input_per_1m": 0.80, "output_per_1m": 4.00}
                    }

                    with patch('run_batch_queries.subprocess.run') as mock_subprocess:
                        mock_subprocess_result = MagicMock()
                        mock_subprocess_result.returncode = 1  # Force failure to check template was rendered
                        mock_subprocess_result.stderr = "Expected error"
                        mock_subprocess.return_value = mock_subprocess_result

                        judge_single_result(result, query_data, config)

                        # Verify the subprocess was called with the rendered template
                        assert mock_subprocess.called
                        call_args = mock_subprocess.call_args[0][0]
                        # The task content is in call_args[3] (the argument after --task flag at index 2)
                        task_content = call_args[3] if len(call_args) > 3 else ""
                        assert "Test query" in task_content

        finally:
            # Clean up
            Path(template_file).unlink()


@patch('run_batch_queries.anthropic.Anthropic')
@patch('run_batch_queries.Path')
def test_anthropic_api_call_debug(mock_path, mock_anthropic):
    """Test to isolate the Anthropic API call issue."""
    sample_result = {
        "query_id": "debug-test",
        "query": "How has monthly gas flow in Texas changed from 2022 to 2024?",
        "success": True,
        "final_answer": "Based on analysis, Texas gas flow increased by 10.08% from 2022 to 2024, with strong growth from 2022 to 2023 (+13.28%) and modest growth from 2023 to 2024 (+1.62%). Monthly flows ranged from 843M to 1,160M units.",
        "execution_time": 47.2
    }

    sample_query_data = {
        "id": "debug-test",
        "query": "How has monthly gas flow in Texas changed from 2022 to 2024?",
        "expected_answer": "Analysis of Texas monthly gas flow trends over 2022-2024",
        "analysis_type": "factual"
    }

    sample_config = {
        "judging": {
            "model": "anthropic:claude-3-5-haiku-20241022",
            "template": "templates/judging_prompt.txt"
        }
    }

    # Mock template file exists
    mock_path.return_value.exists.return_value = True
    mock_path.return_value.read_text.return_value = "Evaluate this query: {{query}}\nExpected: {{expected_answer}}\nActual: {{actual_answer}}"

    # Mock Anthropic client and response
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client

    # Create a mock response that matches the Anthropic API structure
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].input = {
        "accuracy_score": 4,
        "explanation": "Good analysis with clear trends",
        "confidence": 0.85
    }
    mock_response.usage.input_tokens = 150
    mock_response.usage.output_tokens = 75

    # Ensure hasattr returns True for the input attribute
    mock_response.content[0].__dict__['input'] = mock_response.content[0].input

    mock_client.messages.create.return_value = mock_response

    # Mock pricing data
    with patch('run_batch_queries.load_pricing_data') as mock_pricing:
        mock_pricing.return_value = {
            "claude-3-5-haiku-20241022": {
                "input_per_1m": 0.80,
                "output_per_1m": 4.00
            }
        }

        # Execute the function
        result = judge_single_result(sample_result, sample_query_data, sample_config)

        # Debug the actual error
        print(f"Judging result: {result}")
        if not result["judging_performed"]:
            print(f"Error: {result.get('error', 'No error message')}")

        # Verify successful execution
        assert result["judging_performed"] is True
        assert result["accuracy_score"] == 4
        assert result["explanation"] == "Good analysis with clear trends"
        assert result["confidence"] == 0.85
        assert "error" not in result

        # Verify the API was called correctly
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args
        assert call_args[1]["model"] == "claude-3-5-haiku-20241022"
        assert call_args[1]["tools"][0]["name"] == "judge_result"


if __name__ == "__main__":
    pytest.main([__file__])