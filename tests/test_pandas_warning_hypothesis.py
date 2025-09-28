"""
Test to prove/disprove hypothesis that pandas FutureWarnings cause PydanticAI tool failures.

This test reproduces the exact conditions from the failing t002 query to see if
pandas warnings are being misinterpreted as tool execution failures.
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pydantic_agent_executor import write_file_and_run_python, WriteRunArgs, AgentState
from pydantic_ai import RunContext


def test_pandas_warning_does_not_cause_tool_failure():
    """
    Test that a script producing pandas FutureWarnings (like the t002 failure)
    executes successfully and is not considered a tool failure by PydanticAI.

    This directly tests the hypothesis that pandas warnings cause retry failures.
    """

    print("\n=== PANDAS WARNING HYPOTHESIS TEST ===")

    # Create a script that produces the EXACT same FutureWarning as the t002 failure
    script_content = '''
import pandas as pd
import numpy as np
import warnings

print("=== PANDAS WARNING REPRODUCTION TEST ===")

# Create test data similar to pipeline data
data = {
    'eff_gas_day': ['2023-01-01'] * 100,
    'state_abb': ['TX'] * 50 + ['CA'] * 50,
    'scheduled_quantity': np.random.randint(0, 1000, 100),
    'receipt_delivery': ['Receipt'] * 60 + ['Delivery'] * 40
}

df = pd.DataFrame(data)
df['eff_gas_day'] = pd.to_datetime(df['eff_gas_day'])

print(f"Created test DataFrame with {len(df)} rows")

# This will produce the EXACT FutureWarning that appeared in t002 failure logs:
# "DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated..."
def calc_net_consumption(group):
    receipts = group[group['receipt_delivery'] == 'Receipt']['scheduled_quantity'].sum()
    deliveries = group[group['receipt_delivery'] == 'Delivery']['scheduled_quantity'].sum()
    return receipts - deliveries

print("Executing groupby.apply that produces FutureWarning...")

# This line produces the exact warning seen in stderr-1759005074-001_scout_analysis.log
result = df.groupby(['eff_gas_day', 'state_abb']).apply(calc_net_consumption)

print(f"Groupby result computed: {len(result)} state-day combinations")
print("Test completed successfully - script ran to completion despite FutureWarning")
print("=== TEST SUCCESS ===")
'''

    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir)

        # Create mock agent state
        mock_state = MagicMock(spec=AgentState)
        mock_state.workspace_dir = workspace_path
        mock_state.log_react_event = MagicMock()

        # Create mock context
        mock_ctx = MagicMock(spec=RunContext)
        mock_ctx.deps = mock_state

        # Create WriteRunArgs
        args = WriteRunArgs(
            file_path="test_pandas_warning.py",
            content=script_content
        )

        print(f"Testing script with {len(script_content)} characters")
        print("Script will produce pandas FutureWarning...")

        # Execute the tool
        start_time = time.time()
        result = write_file_and_run_python(mock_ctx, args)
        duration = time.time() - start_time

        print(f"Tool execution completed in {duration:.2f}s")
        print(f"Tool result type: {type(result)}")

        # Analyze the result
        print(f"\n=== TOOL EXECUTION ANALYSIS ===")
        print(f"Success: {result.success}")
        print(f"Exit code: {result.exit_code}")
        print(f"Output length: {len(result.stdout_log)} characters")
        print(f"Error length: {len(result.stderr_log or '')} characters")

        if result.stderr_log:
            print(f"Stderr content preview: {result.stderr_log[:200]}...")
            has_future_warning = "FutureWarning" in result.stderr_log
            print(f"Contains FutureWarning: {has_future_warning}")

        if result.stdout_log:
            print(f"Stdout content preview: {result.stdout_log[:200]}...")
            has_success_marker = "TEST SUCCESS" in result.stdout_log
            print(f"Contains success marker: {has_success_marker}")

        # Check log events
        print(f"\n=== REACT EVENT LOG ANALYSIS ===")
        log_calls = mock_state.log_react_event.call_args_list
        print(f"Total log events: {len(log_calls)}")

        for i, call in enumerate(log_calls):
            event_type = call[0][0]
            event_data = call[0][1]
            print(f"Event {i+1}: {event_type}")
            if 'success' in event_data:
                print(f"  Success: {event_data['success']}")

        # THE CRITICAL TEST: Did the tool succeed despite pandas warnings?
        print(f"\n=== HYPOTHESIS TEST RESULTS ===")

        if result.success and result.exit_code == 0:
            print("✅ HYPOTHESIS DISPROVEN: Tool succeeded despite pandas FutureWarning")
            print("   The pandas warning is NOT causing PydanticAI tool failures")
            print("   The retry failures must have a different root cause")
        else:
            print("❌ HYPOTHESIS CONFIRMED: Tool failed due to pandas FutureWarning")
            print("   PydanticAI is incorrectly treating warnings as failures")
            print(f"   Exit code: {result.exit_code}")
            print(f"   Success flag: {result.success}")

        # Verify that the script actually ran to completion
        script_completed = "TEST SUCCESS" in result.stdout_log if result.stdout_log else False
        warning_present = "FutureWarning" in result.stderr_log if result.stderr_log else False

        print(f"\n=== VERIFICATION ===")
        print(f"Script completed successfully: {script_completed}")
        print(f"FutureWarning present in stderr: {warning_present}")

        # Assert the core hypothesis
        if warning_present and script_completed:
            # If warning is present AND script completed, tool should still succeed
            assert result.success, "Tool should succeed when script completes despite warnings"
            assert result.exit_code == 0, "Exit code should be 0 when script completes successfully"
            print("✅ Test proves pandas warnings do NOT cause tool failures")
        else:
            print("⚠️  Test conditions not met - need both warning and completion")

        return {
            'tool_success': result.success,
            'exit_code': result.exit_code,
            'script_completed': script_completed,
            'warning_present': warning_present,
            'hypothesis_confirmed': warning_present and not result.success
        }


def test_tool_failure_detection_mechanism():
    """
    Test what actually causes our write_file_and_run_python tool to report failures.

    This test systematically checks different failure conditions to understand
    the real cause of the retry limit errors.
    """

    print("\n=== TOOL FAILURE DETECTION TEST ===")

    test_cases = [
        {
            'name': 'successful_script',
            'script': 'print("Success")\nexit(0)',
            'expected_success': True
        },
        {
            'name': 'script_with_warnings',
            'script': '''
import warnings
warnings.warn("This is a test warning", FutureWarning)
print("Script completed despite warning")
exit(0)
''',
            'expected_success': True
        },
        {
            'name': 'script_with_explicit_failure',
            'script': 'print("This will fail")\nexit(1)',
            'expected_success': False
        },
        {
            'name': 'script_with_exception',
            'script': 'print("About to raise exception")\nraise ValueError("Test exception")',
            'expected_success': False
        }
    ]

    results = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir)

        for test_case in test_cases:
            print(f"\n--- Testing: {test_case['name']} ---")

            # Create mock agent state
            mock_state = MagicMock(spec=AgentState)
            mock_state.workspace_dir = workspace_path
            mock_state.log_react_event = MagicMock()

            # Create mock context
            mock_ctx = MagicMock(spec=RunContext)
            mock_ctx.deps = mock_state

            # Create WriteRunArgs
            args = WriteRunArgs(
                file_path=f"{test_case['name']}.py",
                content=test_case['script']
            )

            # Execute the tool
            result = write_file_and_run_python(mock_ctx, args)

            print(f"Success: {result.success}")
            print(f"Exit code: {result.exit_code}")
            print(f"Expected success: {test_case['expected_success']}")

            # Check if result matches expectation
            matches_expectation = result.success == test_case['expected_success']
            print(f"Matches expectation: {matches_expectation}")

            results[test_case['name']] = {
                'actual_success': result.success,
                'expected_success': test_case['expected_success'],
                'exit_code': result.exit_code,
                'matches_expectation': matches_expectation
            }

            # Assert the expectation
            assert result.success == test_case['expected_success'], \
                f"Test {test_case['name']} failed: expected {test_case['expected_success']}, got {result.success}"

    print(f"\n=== FAILURE DETECTION SUMMARY ===")
    for name, data in results.items():
        status = "✅ PASS" if data['matches_expectation'] else "❌ FAIL"
        print(f"{status} {name}: success={data['actual_success']}, exit_code={data['exit_code']}")

    return results


if __name__ == "__main__":
    # Run the tests
    print("Testing pandas warning hypothesis...")
    test_pandas_warning_does_not_cause_tool_failure()

    print("\nTesting tool failure detection...")
    test_tool_failure_detection_mechanism()

    print("\n=== ALL TESTS COMPLETED ===")