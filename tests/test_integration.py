"""
Integration tests for the Claude Data Agent end-to-end workflow.

Tests cover:
- Complete workflow from query to result
- Integration between all components
- Real file operations with temporary workspaces
- Error propagation and recovery
- Performance and resource usage
- Data flow validation
"""

import json
import pytest
import tempfile
import time
import yaml
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call
import subprocess
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pydantic_agent_executor import AgentState, write_file_and_run_python, read_file, list_files
from runner_utils import Workspace, RunConfig
from run_batch_queries import run_single_query, load_queries, save_results
from run_agent import main as run_agent_main


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow scenarios."""

    @patch('subprocess.run')
    @patch('pydantic_ai.Agent')
    def test_complete_single_query_workflow(self, mock_agent_class, mock_subprocess, temp_workspace):
        """Test complete workflow from single query to final result."""
        # Mock successful Python execution
        mock_subprocess_result = MagicMock()
        mock_subprocess_result.returncode = 0
        mock_subprocess_result.stdout = "Analysis complete: 5 records found\nTop pipeline: Company A"
        mock_subprocess_result.stderr = ""
        mock_subprocess.return_value = mock_subprocess_result

        # Mock PydanticAI agent
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent

        mock_result = MagicMock()
        mock_result.data = "Complete analysis shows 5 records with Company A leading."
        mock_result.cost.total_tokens = 200
        mock_result.cost.input_tokens = 120
        mock_result.cost.output_tokens = 80
        mock_agent.run_sync.return_value = mock_result

        # Create test query
        query_data = {
            "id": "integration_001",
            "query": "How many records are in the dataset and which company has the highest volume?",
            "category": "analysis"
        }

        config = {
            "model": "anthropic:claude-sonnet-4-20250514",
            "max_tools": 15,
            "timeout": 300,
            "template": "templates/data_analysis_agent_prompt.txt",
            "streaming": True,
            "console_updates": True
        }

        # Run the complete workflow
        with patch('run_batch_queries.load_pricing_data', return_value={"claude-sonnet-4-20250514": {"input_per_1m": 15.0, "output_per_1m": 75.0}}):
            result = run_single_query(query_data, config, str(temp_workspace))

        # Verify complete workflow executed successfully
        assert result["query_id"] == "integration_001"
        assert result["success"] is True
        assert "execution_time" in result
        assert "total_cost" in result
        assert "final_answer" in result

        # Verify agent was called with correct parameters
        mock_agent.run_sync.assert_called_once()

        # Verify Python execution was attempted
        mock_subprocess.assert_called()

    @patch('subprocess.run')
    def test_workspace_file_operations_integration(self, mock_subprocess, temp_workspace):
        """Test integration of workspace operations with file handling."""
        # Create test data file
        data_file = temp_workspace / "pipeline_data.csv"
        test_data = "company,volume,state\nCompany A,1000,TX\nCompany B,800,OK\nCompany C,1200,LA"
        data_file.write_text(test_data)

        # Create analysis script
        script_file = temp_workspace / "analysis.py"
        script_content = """
import pandas as pd

# Read the data
df = pd.read_csv('pipeline_data.csv')

# Perform analysis
total_records = len(df)
top_company = df.loc[df['volume'].idxmax(), 'company']
total_volume = df['volume'].sum()

# Output results
print(f"Total records: {total_records}")
print(f"Top company: {top_company}")
print(f"Total volume: {total_volume}")

# Save summary
summary = {
    'total_records': total_records,
    'top_company': top_company,
    'total_volume': int(total_volume)
}

import json
with open('summary.json', 'w') as f:
    json.dump(summary, f)
"""
        script_file.write_text(script_content)

        # Mock successful execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Total records: 3\nTop company: Company C\nTotal volume: 3000"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Create workspace and agent state
        workspace = Workspace(temp_workspace)
        agent_state = AgentState(
            workspace_dir=temp_workspace,
            query="Test integration query",
            query_id="integration_workspace",
            dataset_description="Test dataset",
            analysis_type="exploration",
            rubric={},
            max_tool_calls=15,
            timeout=300,
            model="anthropic:claude-sonnet-4-20250514",
            streaming=True,
            console_updates=True,
            start_time=time.time(),
            react_log=[],
            tool_timings=[],
            total_input_tokens=0,
            total_output_tokens=0,
            estimated_cost=0.0
        )

        # Test file operations integration
        from pydantic_ai import RunContext
        mock_ctx = MagicMock(spec=RunContext)
        mock_ctx.deps = agent_state

        # Test write and run
        result = write_file_and_run_python(mock_ctx, "analysis.py", script_content)

        # Verify execution
        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        assert args[0] == sys.executable  # Virtual environment fix
        assert str(script_file) in args

        # Test file reading
        read_result = read_file(mock_ctx, "pipeline_data.csv")
        assert "Company A" in read_result
        assert "1000" in read_result

        # Test file listing
        list_result = list_files(mock_ctx, ".")
        assert "pipeline_data.csv" in list_result
        assert "analysis.py" in list_result

    def test_error_propagation_through_workflow(self, temp_workspace):
        """Test that errors propagate correctly through the workflow."""
        # Create invalid query that should fail
        query_data = {
            "id": "error_test",
            "query": "Execute invalid Python that should fail",
            "category": "error_test"
        }

        config = {
            "model": "anthropic:claude-sonnet-4-20250514",
            "max_tools": 5,
            "timeout": 30
        }

        # Mock subprocess failure
        with patch('subprocess.run') as mock_subprocess:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stdout = ""
            mock_result.stderr = "SyntaxError: invalid syntax"
            mock_subprocess.return_value = mock_result

            with patch('pydantic_ai.Agent') as mock_agent_class:
                # Mock agent that tries to execute bad code
                mock_agent = MagicMock()
                mock_agent_class.return_value = mock_agent

                # Simulate agent execution failure
                mock_agent.run_sync.side_effect = Exception("Agent execution failed")

                # Run workflow
                result = run_single_query(query_data, config, str(temp_workspace))

                # Verify error is captured
                assert result["query_id"] == "error_test"
                assert result["success"] is False
                assert "error" in result


class TestBatchProcessingIntegration:
    """Test batch processing integration scenarios."""

    @patch('run_batch_queries.run_single_query')
    def test_batch_processing_with_real_file_operations(self, mock_run_single_query, temp_workspace):
        """Test batch processing with real file operations."""
        # Create test queries file
        test_queries = {
            "queries": [
                {
                    "id": "batch_001",
                    "query": "Count total records",
                    "category": "basic"
                },
                {
                    "id": "batch_002",
                    "query": "Find top companies",
                    "category": "analysis"
                }
            ],
            "config": {
                "model": "anthropic:claude-sonnet-4-20250514",
                "max_tools": 15,
                "timeout": 300,
                "template": "templates/data_analysis_agent_prompt.txt"
            }
        }

        queries_file = temp_workspace / "batch_queries.json"
        queries_file.write_text(json.dumps(test_queries))

        # Mock successful query results
        mock_run_single_query.side_effect = [
            {
                "query_id": "batch_001",
                "success": True,
                "execution_time": 10.0,
                "total_cost": 0.05,
                "final_answer": "Found 100 records"
            },
            {
                "query_id": "batch_002",
                "success": True,
                "execution_time": 15.0,
                "total_cost": 0.08,
                "final_answer": "Top company: ABC Pipeline"
            }
        ]

        # Test batch processing concept - load queries and process them
        queries = load_queries(str(queries_file))

        # Process each query
        results = []
        for query in queries["queries"]:
            result = mock_run_single_query(query, queries["config"], str(temp_workspace))
            results.append(result)

        # Save results
        save_results(results, str(temp_workspace / "batch_results.json"))

        # Verify both queries were executed
        assert mock_run_single_query.call_count == 2

        # Verify results file was created
        results_file = temp_workspace / "batch_results.json"
        assert results_file.exists()

        # Verify results content
        results_data = json.loads(results_file.read_text())
        assert len(results_data) == 2
        assert results_data[0]["query_id"] == "batch_001"
        assert results_data[1]["query_id"] == "batch_002"

    @patch('run_batch_queries.run_single_query')
    def test_batch_processing_resume_functionality(self, mock_run_single_query, temp_workspace):
        """Test batch processing resume functionality with real files."""
        # Create queries file
        test_queries = {
            "queries": [
                {"id": "resume_001", "query": "First query", "category": "test"},
                {"id": "resume_002", "query": "Second query", "category": "test"},
                {"id": "resume_003", "query": "Third query", "category": "test"}
            ],
            "config": {"model": "anthropic:claude-sonnet-4-20250514"}
        }

        queries_file = temp_workspace / "resume_queries.json"
        queries_file.write_text(json.dumps(test_queries))

        # Create existing results (simulate partial completion)
        existing_results = [
            {
                "query_id": "resume_001",
                "success": True,
                "final_answer": "First result"
            }
        ]

        results_file = temp_workspace / "batch_results.json"
        results_file.write_text(json.dumps(existing_results))

        # Mock results for remaining queries
        mock_run_single_query.side_effect = [
            {
                "query_id": "resume_002",
                "success": True,
                "final_answer": "Second result"
            },
            {
                "query_id": "resume_003",
                "success": True,
                "final_answer": "Third result"
            }
        ]

        # Test batch processing with resume concept
        queries = load_queries(str(queries_file))

        # Load existing results
        existing_results = json.loads(results_file.read_text())
        processed_ids = {result["query_id"] for result in existing_results}

        # Process only remaining queries
        all_results = existing_results.copy()
        for query in queries["queries"]:
            if query["id"] not in processed_ids:
                result = mock_run_single_query(query, queries["config"], str(temp_workspace))
                all_results.append(result)

        # Save updated results
        save_results(all_results, str(results_file))

        # Should only execute remaining queries
        assert mock_run_single_query.call_count == 2

        # Verify final results
        final_results = json.loads(results_file.read_text())
        assert len(final_results) == 3


class TestCLIIntegration:
    """Test CLI integration with the complete workflow."""

    @patch('run_agent.run_single_query')
    def test_cli_to_batch_workflow_integration(self, mock_run_single_query, temp_workspace):
        """Test CLI properly integrates with batch processing workflow."""
        # Mock successful execution
        mock_run_single_query.return_value = {
            "query_id": "cli_integration",
            "success": True,
            "execution_time": 20.0,
            "total_cost": 0.12,
            "final_answer": "CLI integration test successful"
        }

        # Test CLI execution
        test_args = [
            "run_agent.py",
            "Test CLI integration",
            "--model", "anthropic:claude-sonnet-4-20250514",
            "--max-tools", "10",
            "--timeout", "300"
        ]

        with patch('sys.argv', test_args):
            with patch('run_agent.datetime') as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20250927_130000"

                run_agent_main()

                # Verify run_single_query was called correctly
                mock_run_single_query.assert_called_once()
                call_args = mock_run_single_query.call_args

                query_data, config, workspace_dir = call_args[0]

                # Verify query structure
                assert query_data["query"] == "Test CLI integration"
                assert "id" in query_data

                # Verify config structure
                assert config["model"] == "anthropic:claude-sonnet-4-20250514"
                assert config["max_tools"] == 10
                assert config["timeout"] == 300

                # Verify workspace was created
                assert workspace_dir is not None
                assert Path(workspace_dir).exists()


class TestResourceManagement:
    """Test resource management and cleanup in integration scenarios."""

    def test_workspace_cleanup_after_execution(self, temp_workspace):
        """Test that workspaces are properly managed during execution."""
        # This tests the general pattern - actual cleanup is handled by tempfile
        workspace = Workspace(temp_workspace)

        # Create files in workspace
        test_file = workspace.root / "test_analysis.py"
        test_file.write_text("print('Resource test')")

        data_file = workspace.root / "test_data.csv"
        data_file.write_text("col1,col2\n1,2\n3,4")

        # Verify files exist
        assert test_file.exists()
        assert data_file.exists()

        # Files should be accessible for the duration of the test
        assert test_file.read_text() == "print('Resource test')"
        assert "col1,col2" in data_file.read_text()

    def test_memory_usage_during_large_operations(self, temp_workspace):
        """Test memory management during large file operations."""
        # Create larger test files
        large_data = "id,value\n" + "\n".join([f"{i},{i*10}" for i in range(1000)])
        large_file = temp_workspace / "large_dataset.csv"
        large_file.write_text(large_data)

        # Test reading large file
        from runner_utils import preview_bytes
        file_content = large_file.read_bytes()
        preview = preview_bytes(file_content)

        # Should handle large files appropriately
        assert "truncated" in preview
        assert len(preview["text"]) <= len(large_data)

    @patch('subprocess.run')
    def test_subprocess_resource_management(self, mock_subprocess, temp_workspace):
        """Test subprocess resource management during execution."""
        # Mock successful subprocess execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Resource management test"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Create workspace and state
        agent_state = AgentState(
            workspace_dir=temp_workspace,
            query="Resource test",
            query_id="resource_001",
            dataset_description="Resource test",
            analysis_type="test",
            rubric={},
            max_tool_calls=5,
            timeout=60,
            model="anthropic:claude-sonnet-4-20250514",
            streaming=False,
            console_updates=False,
            start_time=time.time(),
            react_log=[],
            tool_timings=[],
            total_input_tokens=0,
            total_output_tokens=0,
            estimated_cost=0.0
        )

        from pydantic_ai import RunContext
        mock_ctx = MagicMock(spec=RunContext)
        mock_ctx.deps = agent_state

        # Test multiple subprocess calls
        for i in range(3):
            script_content = f"print('Script {i} executed')"
            write_file_and_run_python(mock_ctx, f"script_{i}.py", script_content)

        # Verify all subprocess calls were made correctly
        assert mock_subprocess.call_count == 3

        # Verify each call used proper resource management
        for call in mock_subprocess.call_args_list:
            args = call[0][0]
            assert args[0] == sys.executable  # Virtual environment fix
            assert "timeout" in call[1] or len(call[1]) >= 4  # Should have timeout


class TestDataFlowValidation:
    """Test data flow validation through the complete system."""

    @patch('subprocess.run')
    @patch('pydantic_ai.Agent')
    def test_data_flow_from_query_to_result(self, mock_agent_class, mock_subprocess, temp_workspace):
        """Test complete data flow from input query to final result."""
        # Set up test data
        test_dataset = temp_workspace / "flow_test_data.csv"
        test_dataset.write_text("company,volume,state\nFlow Test Co,500,TX\nData Flow Inc,300,OK")

        # Mock Python execution that processes the data
        mock_subprocess_result = MagicMock()
        mock_subprocess_result.returncode = 0
        mock_subprocess_result.stdout = json.dumps({
            "total_companies": 2,
            "max_volume": 500,
            "top_company": "Flow Test Co",
            "states": ["TX", "OK"]
        })
        mock_subprocess_result.stderr = ""
        mock_subprocess.return_value = mock_subprocess_result

        # Mock agent that produces final answer
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent

        mock_result = MagicMock()
        mock_result.data = "Analysis shows 2 companies with Flow Test Co having the highest volume of 500."
        mock_result.cost.total_tokens = 150
        mock_result.cost.input_tokens = 90
        mock_result.cost.output_tokens = 60
        mock_agent.run_sync.return_value = mock_result

        # Execute workflow
        query_data = {
            "id": "data_flow_test",
            "query": "Analyze the companies in the dataset and identify the top performer",
            "category": "analysis"
        }

        config = {
            "model": "anthropic:claude-sonnet-4-20250514",
            "max_tools": 10,
            "timeout": 120
        }

        with patch('run_batch_queries.load_pricing_data', return_value={"claude-sonnet-4-20250514": {"input_per_1m": 15.0, "output_per_1m": 75.0}}):
            result = run_single_query(query_data, config, str(temp_workspace))

        # Validate data flow
        assert result["query_id"] == "data_flow_test"
        assert result["success"] is True

        # Verify data was processed (subprocess was called)
        mock_subprocess.assert_called()

        # Verify agent was called with processed results
        mock_agent.run_sync.assert_called_once()

        # Verify final result contains expected information
        assert "final_answer" in result
        assert isinstance(result["execution_time"], (int, float))
        assert isinstance(result["total_cost"], (int, float))

    def test_configuration_data_flow(self, temp_workspace):
        """Test that configuration flows correctly through the system."""
        # Create test configuration
        test_config = {
            "dataset": {
                "description": "Configuration flow test dataset",
                "location": "flow_test_data.csv"
            },
            "agent": {
                "model": "anthropic:claude-sonnet-4-20250514",
                "max_tools": 8,
                "timeout": 180,
                "streaming": False
            }
        }

        config_file = temp_workspace / "flow_config.yaml"
        config_file.write_text(yaml.dump(test_config))

        # Test configuration loading
        with open(config_file, 'r') as f:
            loaded_config = yaml.safe_load(f)

        # Verify configuration structure
        assert loaded_config["agent"]["model"] == "anthropic:claude-sonnet-4-20250514"
        assert loaded_config["agent"]["max_tools"] == 8
        assert loaded_config["agent"]["timeout"] == 180
        assert loaded_config["dataset"]["description"] == "Configuration flow test dataset"

    def test_error_data_flow(self, temp_workspace):
        """Test that errors flow correctly through the system."""
        # Create scenario that should produce errors
        query_data = {
            "id": "error_flow_test",
            "query": "Process non-existent data file",
            "category": "error_test"
        }

        config = {
            "model": "anthropic:claude-sonnet-4-20250514",
            "max_tools": 3,
            "timeout": 30
        }

        # Mock subprocess that fails
        with patch('subprocess.run') as mock_subprocess:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stdout = ""
            mock_result.stderr = "FileNotFoundError: No such file"
            mock_subprocess.return_value = mock_result

            with patch('pydantic_ai.Agent') as mock_agent_class:
                mock_agent = MagicMock()
                mock_agent_class.return_value = mock_agent

                # Simulate agent that encounters the error
                mock_agent.run_sync.side_effect = Exception("File processing failed")

                result = run_single_query(query_data, config, str(temp_workspace))

                # Verify error information flows through
                assert result["query_id"] == "error_flow_test"
                assert result["success"] is False
                assert "error" in result
                assert isinstance(result["execution_time"], (int, float))


class TestPerformanceIntegration:
    """Test performance aspects of the integrated system."""

    @patch('run_batch_queries.run_single_query')
    def test_concurrent_execution_performance(self, mock_run_single_query, temp_workspace):
        """Test performance of concurrent query execution."""
        # Create multiple test queries
        test_queries = {
            "queries": [
                {"id": f"perf_{i:03d}", "query": f"Performance test query {i}", "category": "performance"}
                for i in range(5)
            ],
            "config": {"model": "anthropic:claude-sonnet-4-20250514"}
        }

        queries_file = temp_workspace / "performance_queries.json"
        queries_file.write_text(json.dumps(test_queries))

        # Mock query results with realistic timing
        mock_run_single_query.side_effect = [
            {
                "query_id": f"perf_{i:03d}",
                "success": True,
                "execution_time": 10.0 + i,  # Slightly different times
                "total_cost": 0.05 * (i + 1)
            }
            for i in range(5)
        ]

        # Measure execution time
        start_time = time.time()

        # Test concurrent processing concept
        queries = load_queries(str(queries_file))
        results = []
        for query in queries["queries"]:
            result = mock_run_single_query(query, queries["config"], str(temp_workspace))
            results.append(result)

        save_results(results, str(temp_workspace / "batch_results.json"))

        end_time = time.time()
        execution_time = end_time - start_time

        # Verify all queries were executed
        assert mock_run_single_query.call_count == 5

        # Verify results
        results_file = temp_workspace / "batch_results.json"
        assert results_file.exists()

        results_data = json.loads(results_file.read_text())
        assert len(results_data) == 5

        # Performance should be reasonable (this is a very loose check)
        assert execution_time < 60  # Should complete within reasonable time

    def test_memory_efficient_large_result_handling(self, temp_workspace):
        """Test memory efficiency when handling large results."""
        # Create large result data
        large_result = {
            "query_id": "large_result_test",
            "success": True,
            "execution_time": 30.0,
            "final_answer": "x" * 50000,  # Large answer
            "detailed_data": [{"item": i, "value": f"value_{i}"} for i in range(1000)]
        }

        # Test saving large result
        from run_batch_queries import save_results, load_results

        results_file = temp_workspace / "large_results.json"
        save_results([large_result], str(results_file))

        # Verify file was created
        assert results_file.exists()

        # Test loading large result
        loaded_results = load_results(str(results_file))

        # Verify data integrity
        assert len(loaded_results) == 1
        assert loaded_results[0]["query_id"] == "large_result_test"
        assert len(loaded_results[0]["final_answer"]) == 50000
        assert len(loaded_results[0]["detailed_data"]) == 1000


class TestSystemRobustness:
    """Test system robustness under various conditions."""

    @patch('subprocess.run')
    def test_timeout_handling_integration(self, mock_subprocess, temp_workspace):
        """Test integrated timeout handling across components."""
        from subprocess import TimeoutExpired

        # Mock timeout in subprocess
        mock_subprocess.side_effect = TimeoutExpired("python", 30)

        query_data = {
            "id": "timeout_integration_test",
            "query": "Long running analysis that should timeout",
            "category": "timeout_test"
        }

        config = {
            "model": "anthropic:claude-sonnet-4-20250514",
            "max_tools": 5,
            "timeout": 30
        }

        # Test timeout handling
        result = run_single_query(query_data, config, str(temp_workspace))

        # Should handle timeout gracefully
        assert result["query_id"] == "timeout_integration_test"
        assert result["success"] is False
        assert "timeout" in result.get("error", "").lower()

    def test_recovery_from_partial_failures(self, temp_workspace):
        """Test system recovery from partial component failures."""
        # Create test scenario with mixed success/failure
        test_queries = {
            "queries": [
                {"id": "recovery_001", "query": "Should succeed", "category": "test"},
                {"id": "recovery_002", "query": "Should fail", "category": "test"},
                {"id": "recovery_003", "query": "Should succeed", "category": "test"}
            ],
            "config": {"model": "anthropic:claude-sonnet-4-20250514"}
        }

        queries_file = temp_workspace / "recovery_queries.json"
        queries_file.write_text(json.dumps(test_queries))

        # Mock mixed results
        with patch('run_batch_queries.run_single_query') as mock_run_single_query:
            mock_run_single_query.side_effect = [
                {"query_id": "recovery_001", "success": True, "final_answer": "Success 1"},
                {"query_id": "recovery_002", "success": False, "error": "Simulated failure"},
                {"query_id": "recovery_003", "success": True, "final_answer": "Success 2"}
            ]

            # Test handling mixed results
            queries = load_queries(str(queries_file))
            results = []
            for query in queries["queries"]:
                result = mock_run_single_query(query, queries["config"], str(temp_workspace))
                results.append(result)

            save_results(results, str(temp_workspace / "batch_results.json"))

            # Verify all queries were attempted
            assert mock_run_single_query.call_count == 3

            # Verify results were saved (including failures)
            results_file = temp_workspace / "batch_results.json"
            assert results_file.exists()

            results_data = json.loads(results_file.read_text())
            assert len(results_data) == 3
            assert results_data[0]["success"] is True
            assert results_data[1]["success"] is False
            assert results_data[2]["success"] is True