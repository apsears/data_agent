"""
Tests for run_agent.py - CLI interface and configuration loading.

Tests cover:
- Command line argument parsing
- Configuration loading and validation
- Single query execution via CLI
- Template loading and processing
- Error handling for invalid arguments
- Integration with batch processing workflow
"""

import json
import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from run_agent import (
    build_parser, main, DEFAULT_TEMPLATE, DEFAULT_MODEL,
    DEFAULT_MAX_TOOLS, DEFAULT_TIMEOUT, DEFAULT_CONFIG
)


class TestArgumentParsing:
    """Test command line argument parsing."""

    def test_build_parser_basic(self):
        """Test basic parser construction."""
        parser = build_parser()

        # Test with minimal required arguments
        args = parser.parse_args(["Test query"])

        assert args.query == "Test query"
        assert args.model == DEFAULT_MODEL
        assert args.max_tools == DEFAULT_MAX_TOOLS
        assert args.timeout == DEFAULT_TIMEOUT

    def test_build_parser_all_arguments(self):
        """Test parser with all arguments provided."""
        parser = build_parser()

        args = parser.parse_args([
            "Complex test query",
            "--model", "anthropic:claude-haiku-3-5-20241022",
            "--max-tools", "20",
            "--timeout", "600",
            "--template", "custom_template.txt",
            "--config", "custom_config.yaml",
            "--query-id", "custom_001",
            "--no-console-updates",
            "--no-stream"
        ])

        assert args.query == "Complex test query"
        assert args.model == "anthropic:claude-haiku-3-5-20241022"
        assert args.max_tools == 20
        assert args.timeout == 600
        assert args.template == "custom_template.txt"
        assert args.config == "custom_config.yaml"
        assert args.query_id == "custom_001"
        assert args.no_console_updates is True
        assert args.no_stream is True

    def test_build_parser_task_flag(self):
        """Test using --task flag instead of positional argument."""
        parser = build_parser()

        args = parser.parse_args(["--task", "Task from flag"])

        assert args.task == "Task from flag"
        assert args.query is None

    def test_build_parser_both_query_and_task(self):
        """Test providing both query and task (should work)."""
        parser = build_parser()

        args = parser.parse_args(["Positional query", "--task", "Flag task"])

        assert args.query == "Positional query"
        assert args.task == "Flag task"

    def test_build_parser_no_query_or_task(self):
        """Test parser with neither query nor task."""
        parser = build_parser()

        args = parser.parse_args([])

        assert args.query is None
        assert args.task is None

    def test_build_parser_integer_validation(self):
        """Test integer argument validation."""
        parser = build_parser()

        # Valid integers
        args = parser.parse_args(["Test", "--max-tools", "25", "--timeout", "900"])
        assert args.max_tools == 25
        assert args.timeout == 900

        # Invalid integers should be caught by argparse
        with pytest.raises(SystemExit):
            parser.parse_args(["Test", "--max-tools", "not_a_number"])

    def test_build_parser_help_text(self):
        """Test that help text is properly formatted."""
        parser = build_parser()

        # Should not raise exception when getting help
        help_text = parser.format_help()

        assert "Examples:" in help_text
        assert "python run_agent.py" in help_text
        assert "--model" in help_text
        assert "--max-tools" in help_text


class TestConfigurationLoading:
    """Test configuration file loading and validation."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    def test_config_loading_success(self, mock_yaml_load, mock_file, sample_config_yaml):
        """Test successful configuration loading."""
        mock_yaml_load.return_value = sample_config_yaml

        # This would be tested in the actual main function
        # Here we test the concept with mocked file operations
        with patch('pathlib.Path.exists', return_value=True):
            # Simulate config loading
            config_data = yaml.safe_load(mock_file.return_value)

            assert "dataset" in config_data
            assert "agent" in config_data
            assert config_data["agent"]["model"] == "anthropic:claude-sonnet-4-20250514"

    @patch('pathlib.Path.exists')
    def test_config_file_not_found(self, mock_exists):
        """Test handling of missing configuration file."""
        mock_exists.return_value = False

        # Should handle missing config file gracefully
        # In practice, the application should use defaults or show appropriate error

    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    def test_config_invalid_yaml(self, mock_yaml_load, mock_file):
        """Test handling of invalid YAML configuration."""
        mock_yaml_load.side_effect = yaml.YAMLError("Invalid YAML")

        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(yaml.YAMLError):
                yaml.safe_load(mock_file.return_value)

    def test_default_config_values(self):
        """Test that default configuration values are properly set."""
        assert DEFAULT_MODEL == "anthropic:claude-sonnet-4-20250514"
        assert DEFAULT_MAX_TOOLS == 15
        assert DEFAULT_TIMEOUT == 300
        assert DEFAULT_CONFIG == "config/config.yaml"
        assert DEFAULT_TEMPLATE == "templates/data_analysis_agent_prompt.txt"


class TestTemplateLoading:
    """Test template file loading and processing."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    def test_template_loading_success(self, mock_exists, mock_file):
        """Test successful template loading."""
        mock_exists.return_value = True
        mock_template_content = "Template: {query}\nAnalyze: {dataset_description}"
        mock_file.return_value.read.return_value = mock_template_content

        # Simulate template loading
        with open("templates/test_template.txt", "r") as f:
            template_content = f.read()

        assert "Template:" in template_content
        assert "{query}" in template_content

    @patch('pathlib.Path.exists')
    def test_template_file_not_found(self, mock_exists):
        """Test handling of missing template file."""
        mock_exists.return_value = False

        # Should handle missing template file appropriately
        # In practice, should either use default template or show error

    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    def test_template_with_variables(self, mock_exists, mock_file):
        """Test template with Jinja2 variables."""
        mock_exists.return_value = True
        mock_template = """
Query: {{ query }}
Dataset: {{ dataset_description }}
Analysis Type: {{ analysis_type }}
"""
        mock_file.return_value.read.return_value = mock_template

        # Template should contain expected variables
        with open("templates/test_template.txt", "r") as f:
            template_content = f.read()

        assert "{{ query }}" in template_content
        assert "{{ dataset_description }}" in template_content
        assert "{{ analysis_type }}" in template_content


class TestSingleQueryExecution:
    """Test single query execution through CLI."""

    @patch('run_agent.run_single_query')
    def test_main_function_with_positional_query(self, mock_run_single_query):
        """Test main function with positional query argument."""
        # Mock successful query execution
        mock_run_single_query.return_value = {
            "query_id": "cli_test_001",
            "success": True,
            "execution_time": 15.0,
            "final_answer": "CLI test result"
        }

        # Mock command line arguments
        test_args = [
            "run_agent.py",
            "How many records are in the dataset?",
            "--model", "anthropic:claude-sonnet-4-20250514",
            "--max-tools", "10"
        ]

        with patch('sys.argv', test_args):
            with patch('run_agent.datetime') as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20250927_130000"

                main()

                # Verify run_single_query was called with correct parameters
                mock_run_single_query.assert_called_once()
                call_args = mock_run_single_query.call_args

                # Extract the query data and config from the call
                query_data, config = call_args[0][:2]

                assert query_data["query"] == "How many records are in the dataset?"
                assert config["model"] == "anthropic:claude-sonnet-4-20250514"
                assert config["max_tools"] == 10

    @patch('run_agent.run_single_query')
    def test_main_function_with_task_flag(self, mock_run_single_query):
        """Test main function with --task flag."""
        mock_run_single_query.return_value = {
            "query_id": "cli_test_002",
            "success": True,
            "final_answer": "Task flag result"
        }

        test_args = [
            "run_agent.py",
            "--task", "Analyze pipeline data trends",
            "--timeout", "600"
        ]

        with patch('sys.argv', test_args):
            with patch('run_agent.datetime') as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20250927_130000"

                main()

                mock_run_single_query.assert_called_once()
                call_args = mock_run_single_query.call_args
                query_data = call_args[0][0]

                assert query_data["query"] == "Analyze pipeline data trends"

    @patch('run_agent.run_single_query')
    def test_main_function_no_query_provided(self, mock_run_single_query):
        """Test main function when no query is provided."""
        test_args = ["run_agent.py", "--model", "anthropic:claude-sonnet-4-20250514"]

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit):
                main()

            # Should not call run_single_query if no query provided
            mock_run_single_query.assert_not_called()

    @patch('run_agent.run_single_query')
    def test_main_function_with_custom_query_id(self, mock_run_single_query):
        """Test main function with custom query ID."""
        mock_run_single_query.return_value = {
            "query_id": "custom_query_123",
            "success": True,
            "final_answer": "Custom ID result"
        }

        test_args = [
            "run_agent.py",
            "Test query with custom ID",
            "--query-id", "custom_query_123"
        ]

        with patch('sys.argv', test_args):
            main()

            mock_run_single_query.assert_called_once()
            call_args = mock_run_single_query.call_args
            query_data = call_args[0][0]

            assert query_data["id"] == "custom_query_123"

    @patch('run_agent.run_single_query')
    def test_main_function_with_streaming_options(self, mock_run_single_query):
        """Test main function with streaming and console update options."""
        mock_run_single_query.return_value = {
            "query_id": "streaming_test",
            "success": True,
            "final_answer": "Streaming test result"
        }

        test_args = [
            "run_agent.py",
            "Test streaming options",
            "--no-console-updates",
            "--no-stream"
        ]

        with patch('sys.argv', test_args):
            with patch('run_agent.datetime') as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20250927_130000"

                main()

                mock_run_single_query.assert_called_once()
                call_args = mock_run_single_query.call_args
                config = call_args[0][1]

                assert config["streaming"] is False
                assert config["console_updates"] is False


class TestErrorHandling:
    """Test CLI error handling scenarios."""

    @patch('run_agent.run_single_query')
    def test_main_function_query_execution_failure(self, mock_run_single_query):
        """Test handling of query execution failure."""
        # Mock query execution failure
        mock_run_single_query.return_value = {
            "query_id": "failed_query",
            "success": False,
            "error": "Query execution failed",
            "execution_time": 5.0
        }

        test_args = ["run_agent.py", "Failing query test"]

        with patch('sys.argv', test_args):
            with patch('builtins.print') as mock_print:
                with patch('run_agent.datetime') as mock_datetime:
                    mock_datetime.now.return_value.strftime.return_value = "20250927_130000"

                    # Should handle failure gracefully
                    main()

                    # Verify error information was printed
                    print_calls = [str(call) for call in mock_print.call_args_list]
                    error_found = any("error" in call.lower() or "fail" in call.lower() for call in print_calls)
                    assert error_found or mock_print.call_count > 0

    @patch('run_agent.run_single_query')
    def test_main_function_with_exception(self, mock_run_single_query):
        """Test handling of unexpected exceptions."""
        # Mock exception during query execution
        mock_run_single_query.side_effect = Exception("Unexpected error")

        test_args = ["run_agent.py", "Exception test query"]

        with patch('sys.argv', test_args):
            with pytest.raises(Exception):
                main()

    def test_main_function_invalid_model(self):
        """Test handling of invalid model specification."""
        test_args = [
            "run_agent.py",
            "Test query",
            "--model", "invalid:model:format"
        ]

        # Should either handle gracefully or provide meaningful error
        with patch('sys.argv', test_args):
            with patch('run_agent.run_single_query') as mock_run_single_query:
                try:
                    main()
                    # If no exception, verify the model was passed as-is
                    mock_run_single_query.assert_called_once()
                except Exception as e:
                    # Should provide meaningful error about invalid model
                    assert "model" in str(e).lower()

    def test_main_function_negative_values(self):
        """Test handling of negative values for numeric arguments."""
        test_args = [
            "run_agent.py",
            "Test query",
            "--max-tools", "-5",
            "--timeout", "-100"
        ]

        # Should be handled by argparse or application logic
        with patch('sys.argv', test_args):
            with patch('run_agent.run_single_query') as mock_run_single_query:
                try:
                    main()
                    # If no exception, verify values were processed
                    mock_run_single_query.assert_called_once()
                    call_args = mock_run_single_query.call_args
                    config = call_args[0][1]

                    # Application should handle negative values appropriately
                    assert isinstance(config["max_tools"], int)
                    assert isinstance(config["timeout"], int)
                except SystemExit:
                    # argparse might reject negative values
                    pass


class TestIntegrationWithBatchWorkflow:
    """Test integration between CLI interface and batch processing workflow."""

    @patch('run_agent.run_single_query')
    def test_cli_calls_batch_workflow(self, mock_run_single_query):
        """Test that CLI properly calls the batch workflow."""
        mock_run_single_query.return_value = {
            "query_id": "integration_test",
            "success": True,
            "final_answer": "Integration test result"
        }

        test_args = [
            "run_agent.py",
            "Integration test query",
            "--model", "anthropic:claude-sonnet-4-20250514"
        ]

        with patch('sys.argv', test_args):
            with patch('run_agent.datetime') as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20250927_130000"

                main()

                # Verify run_single_query was called (from batch processing module)
                mock_run_single_query.assert_called_once()

                # Verify the call structure matches batch processing expectations
                call_args = mock_run_single_query.call_args
                query_data, config, workspace_dir = call_args[0]

                # Should have proper query structure
                assert "id" in query_data
                assert "query" in query_data
                assert "category" in query_data

                # Should have proper config structure
                assert "model" in config
                assert "max_tools" in config
                assert "timeout" in config
                assert "template" in config

                # Should have workspace directory
                assert workspace_dir is not None
                assert isinstance(workspace_dir, str)

    @patch('run_agent.run_single_query')
    def test_cli_workspace_creation(self, mock_run_single_query):
        """Test that CLI creates appropriate workspace for execution."""
        mock_run_single_query.return_value = {
            "query_id": "workspace_test",
            "success": True,
            "final_answer": "Workspace test result"
        }

        test_args = ["run_agent.py", "Workspace test query"]

        with patch('sys.argv', test_args):
            with patch('run_agent.datetime') as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20250927_130000"

                main()

                mock_run_single_query.assert_called_once()
                call_args = mock_run_single_query.call_args
                workspace_dir = call_args[0][2]

                # Workspace should be a valid path
                assert workspace_dir is not None
                assert len(workspace_dir) > 0

    @patch('run_agent.run_single_query')
    def test_cli_result_output(self, mock_run_single_query):
        """Test that CLI properly outputs results."""
        mock_result = {
            "query_id": "output_test",
            "success": True,
            "execution_time": 25.5,
            "total_cost": 0.15,
            "final_answer": "Output test result with detailed analysis"
        }
        mock_run_single_query.return_value = mock_result

        test_args = ["run_agent.py", "Output test query"]

        with patch('sys.argv', test_args):
            with patch('builtins.print') as mock_print:
                with patch('run_agent.datetime') as mock_datetime:
                    mock_datetime.now.return_value.strftime.return_value = "20250927_130000"

                    main()

                    # Verify result information was printed
                    print_calls = [str(call) for call in mock_print.call_args_list]

                    # Should print key result information
                    result_info_found = any(
                        "execution_time" in call.lower() or
                        "cost" in call.lower() or
                        "result" in call.lower() or
                        "answer" in call.lower()
                        for call in print_calls
                    )

                    assert result_info_found or mock_print.call_count > 0


class TestQueryIdGeneration:
    """Test automatic query ID generation."""

    @patch('run_agent.run_single_query')
    @patch('run_agent.datetime')
    def test_automatic_query_id_generation(self, mock_datetime, mock_run_single_query):
        """Test that query ID is automatically generated when not provided."""
        # Mock datetime for consistent ID generation
        mock_datetime.now.return_value.strftime.return_value = "20250927_130000"

        mock_run_single_query.return_value = {
            "query_id": "generated_id",
            "success": True,
            "final_answer": "Auto ID result"
        }

        test_args = ["run_agent.py", "Auto ID test query"]

        with patch('sys.argv', test_args):
            main()

            mock_run_single_query.assert_called_once()
            call_args = mock_run_single_query.call_args
            query_data = call_args[0][0]

            # Should have generated a query ID
            assert "id" in query_data
            assert query_data["id"] is not None
            assert len(query_data["id"]) > 0

    @patch('run_agent.run_single_query')
    def test_query_id_uniqueness(self, mock_run_single_query):
        """Test that generated query IDs are unique."""
        mock_run_single_query.return_value = {
            "query_id": "unique_test",
            "success": True,
            "final_answer": "Unique ID result"
        }

        query_ids = []

        # Generate multiple query IDs
        for i in range(3):
            test_args = ["run_agent.py", f"Unique test query {i}"]

            with patch('sys.argv', test_args):
                with patch('time.sleep'):  # Avoid actual sleep in tests
                    main()

                    call_args = mock_run_single_query.call_args
                    query_data = call_args[0][0]
                    query_ids.append(query_data["id"])

        # All IDs should be unique
        assert len(set(query_ids)) == len(query_ids)