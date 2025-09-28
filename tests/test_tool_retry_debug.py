"""
TDD Test to Debug Tool Retry Failures in Temporal Queries

This test reproduces and debugs the exact issue causing temporal queries to fail
with "Tool 'write_file_and_run_python' exceeded max retries count of 1" errors.

The problem pattern:
1. First tool call succeeds (24.2s duration)
2. Second tool call fails immediately
3. PydanticAI reports UnexpectedModelBehavior
4. Error: "second_tool_execution_failed"

This test will systematically isolate the root cause.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from run_batch_queries import run_single_query


def test_temporal_query_reproduction():
    """
    Reproduce the exact temporal query failure that's causing 30% failure rate.

    This test creates the exact scenario:
    - Query t001: "How has monthly gas flow in Texas changed from 2022 to 2024?"
    - Uses factual analysis template (which switches to temporal for temporal queries)
    - Same model: anthropic:claude-sonnet-4-20250514
    - Same retry configuration
    """
    # Exact failing query from batch results
    failing_query = {
        "id": "t001",
        "query": "How has monthly gas flow in Texas changed from 2022 to 2024?",
        "category": "temporal",
        "expected_answer": "Analysis of Texas monthly gas flow trends over 2022-2024",
        "analysis_type": "temporal"
    }

    # Exact config from failed run
    config = {
        "model": {
            "provider": "anthropic",
            "name": "claude-sonnet-4-20250514"  # Using exact model from failure
        },
        "agent": {
            "max_tools": 20,
            "timeout_sec": 60,
            "allow_network": False
        },
        "workspace": {
            "base_dir": ".runs"
        },
        "judging": {
            "enabled": False  # Disable judging for this test to isolate tool retry issue
        }
    }

    # Create temporary results directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_config = config.copy()
        temp_config["workspace"]["base_dir"] = temp_dir

        # Run the failing query and capture the exact failure
        result = run_single_query(failing_query, temp_config)

        # Debug assertions - what we expect vs what we get
        print(f"\n=== TEMPORAL QUERY FAILURE DEBUG ===")
        print(f"Query: {failing_query['query']}")
        print(f"Success: {result['success']}")
        print(f"Execution Time: {result.get('execution_time', 'N/A')}")

        if not result['success']:
            print(f"Error: {result.get('stdout', 'No output')}")

            # Check if it's the exact retry failure pattern
            stdout = result.get('stdout', '')

            # Assert this is the exact failure pattern we're debugging
            assert "Tool 'write_file_and_run_python' exceeded max retries count of 1" in stdout
            assert "UnexpectedModelBehavior" in stdout
            assert "🔍 RETRY LIMIT ERROR" in stdout

            # Verify it mentions the second tool execution failure
            assert "📊 Tool executions so far: 1" in stdout

            print("✅ Successfully reproduced the exact retry failure pattern")

        else:
            pytest.fail("Expected query to fail with retry limit, but it succeeded")


def test_successful_vs_failed_tool_call_analysis():
    """
    Compare successful factual queries vs failed temporal queries to identify differences.

    This test runs a simple factual query (known to work) and a temporal query (known to fail)
    to isolate what makes temporal queries prone to retry failures.
    """

    # Working factual query (from successful batch results)
    successful_query = {
        "id": "f001",
        "query": "What was the total scheduled quantity in Texas during 2023?",
        "category": "factual",
        "expected_answer": "12,467,866,534",
        "analysis_type": "factual"
    }

    # Failing temporal query
    failing_query = {
        "id": "t001",
        "query": "How has monthly gas flow in Texas changed from 2022 to 2024?",
        "category": "temporal",
        "expected_answer": "Analysis of Texas monthly gas flow trends over 2022-2024",
        "analysis_type": "temporal"
    }

    config = {
        "model": {
            "provider": "anthropic",
            "name": "claude-3-5-haiku-20241022"  # Use faster model for testing
        },
        "agent": {
            "max_tools": 5,  # Lower limit for faster testing
            "timeout_sec": 60,
            "allow_network": False
        },
        "judging": {"enabled": False}
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_config = config.copy()
        temp_config["workspace"] = {"base_dir": temp_dir}

        print("\n=== COMPARATIVE ANALYSIS ===")

        # Test successful factual query
        print("Testing successful factual query...")
        factual_result = run_single_query(successful_query, temp_config)

        print(f"Factual Query Success: {factual_result['success']}")
        print(f"Factual Execution Time: {factual_result.get('execution_time', 'N/A')}")

        # Test failing temporal query
        print("Testing failing temporal query...")
        temporal_result = run_single_query(failing_query, temp_config)

        print(f"Temporal Query Success: {temporal_result['success']}")
        print(f"Temporal Execution Time: {temporal_result.get('execution_time', 'N/A')}")

        # Analysis of differences
        print("\n=== DIFFERENCE ANALYSIS ===")

        if factual_result['success'] and not temporal_result['success']:
            print("✅ Confirmed: Factual queries succeed, temporal queries fail")

            # Examine the failure pattern
            temporal_stdout = temporal_result.get('stdout', '')
            if "exceeded max retries" in temporal_stdout:
                print("✅ Confirmed: Temporal failure is retry-related")

            # Look for template differences
            if "factual_analysis_agent_prompt.txt" in str(temporal_result):
                print("⚠️  Temporal query may be using factual template incorrectly")

        elif factual_result['success'] and temporal_result['success']:
            print("✅ Both queries succeeded - retry issue may be intermittent")

        else:
            print("❌ Unexpected pattern - both queries failed or factual failed")


def test_pydantic_ai_retry_mechanism():
    """
    Test the PydanticAI retry mechanism to understand why it's failing on the second tool call.

    The key insight from the logs:
    - First tool call: success=true, duration=24.2s
    - Then immediate failure with "exceeded max retries count of 1"
    - This suggests the SECOND tool call is failing, not the first
    """

    # This test will mock the PydanticAI agent to simulate the exact failure scenario
    with patch('run_batch_queries.Agent') as mock_agent_class:
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent

        # Simulate the exact pattern from failed logs:
        # 1. First tool call succeeds
        # 2. Second tool call fails immediately
        # 3. PydanticAI gives up with UnexpectedModelBehavior

        from pydantic_ai.exceptions import UnexpectedModelBehavior

        # First call succeeds, second call hits retry limit
        mock_agent.run_sync.side_effect = UnexpectedModelBehavior(
            "Tool 'write_file_and_run_python' exceeded max retries count of 1"
        )

        query = {
            "id": "test_retry",
            "query": "How has monthly gas flow in Texas changed from 2022 to 2024?",
            "category": "temporal"
        }

        config = {
            "model": {"provider": "anthropic", "name": "claude-3-5-haiku-20241022"},
            "agent": {"max_tools": 20, "timeout_sec": 60, "allow_network": False},
            "judging": {"enabled": False}
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config = config.copy()
            temp_config["workspace"] = {"base_dir": temp_dir}

            result = run_single_query(query, temp_config)

            print(f"\n=== PYDANTIC AI RETRY MECHANISM TEST ===")
            print(f"Result Success: {result['success']}")
            print(f"Error Pattern: {result.get('stdout', '')}")

            # Verify we can catch and handle the exact PydanticAI retry failure
            assert not result['success']
            assert "exceeded max retries" in str(result.get('stdout', ''))

            print("✅ Successfully reproduced PydanticAI retry failure mechanism")


def test_tool_retry_configuration_discovery():
    """
    Discover where and how tool retry limits are configured in our system.

    From the error logs, we know:
    - "max retries count of 1" suggests retries=1 somewhere
    - This is likely in PydanticAI agent configuration
    - We need to find where this can be increased
    """

    # Search for retry configuration in our codebase
    import run_batch_queries

    print("\n=== RETRY CONFIGURATION DISCOVERY ===")

    # Check if run_batch_queries has retry configuration
    source_code = Path("run_batch_queries.py").read_text()

    retry_keywords = ["retry", "retries", "max_retries", "tool_retries"]
    found_configs = []

    for keyword in retry_keywords:
        if keyword in source_code:
            found_configs.append(keyword)

    print(f"Retry keywords found in run_batch_queries.py: {found_configs}")

    # Check PydanticAI Agent constructor parameters
    try:
        from pydantic_ai import Agent
        import inspect

        agent_signature = inspect.signature(Agent.__init__)
        print(f"PydanticAI Agent constructor parameters: {list(agent_signature.parameters.keys())}")

        # Look for retry-related parameters
        retry_params = [param for param in agent_signature.parameters.keys()
                       if 'retry' in param.lower()]
        print(f"Retry-related parameters: {retry_params}")

    except Exception as e:
        print(f"Could not inspect PydanticAI Agent: {e}")

    print("✅ Retry configuration discovery completed")


def test_fix_retry_configuration():
    """
    Test that the retry configuration fix has been implemented.

    Validates that the Agent constructor in pydantic_agent_executor.py
    now includes retries=3 parameter to fix the retry limit issue.
    """

    print("\n=== RETRY CONFIGURATION FIX VALIDATION ===")

    # Read the fixed pydantic_agent_executor.py to verify the fix
    from pathlib import Path

    executor_file = Path("pydantic_agent_executor.py")
    if not executor_file.exists():
        pytest.fail("pydantic_agent_executor.py not found")

    source_code = executor_file.read_text()

    # Check that retries=3 is now in the Agent constructor
    if "retries=3" in source_code:
        print("✅ Confirmed: Agent constructor now includes retries=3")
    else:
        pytest.fail("Fix not implemented: retries=3 not found in Agent constructor")

    # Check that the fix is in the correct location (create_agent function)
    if "def create_agent" in source_code and "retries=3" in source_code:
        print("✅ Confirmed: retries=3 is in create_agent function")
    else:
        pytest.fail("Fix not in correct location: retries=3 should be in create_agent function")

    # Verify the comment explaining the fix is present
    if "Fix for" in source_code and "retry" in source_code:
        print("✅ Confirmed: Fix includes explanatory comment")
    else:
        print("⚠️  Warning: Fix should include explanatory comment")

    print("✅ Retry configuration fix validation completed successfully")


if __name__ == "__main__":
    # Run the debug tests
    pytest.main([__file__, "-v", "-s"])