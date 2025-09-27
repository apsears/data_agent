"""
Shared pytest fixtures for Claude Data Agent tests.
"""

import json
import os
import tempfile
import pytest
import yaml
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from typing import Dict, Any, List

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pydantic_agent_executor import AgentState, ToolTiming
from runner_utils import Workspace, RunConfig


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        yield workspace


@pytest.fixture
def sample_agent_state(temp_workspace):
    """Create a sample AgentState for testing."""
    return AgentState(
        workspace_dir=temp_workspace,
        query="Test query about pipeline data",
        query_id="test_001",
        dataset_description="Test dataset description",
        analysis_type="exploration",
        rubric={
            "criteria": ["accuracy", "completeness"],
            "scoring": {"max_score": 10}
        },
        max_tool_calls=15,
        timeout=300,
        model="anthropic:claude-sonnet-4-20250514",
        streaming=True,
        console_updates=True,
        start_time=1234567890.0,
        react_log=[],
        tool_timings=[],
        total_input_tokens=0,
        total_output_tokens=0,
        estimated_cost=0.0
    )


@pytest.fixture
def sample_workspace():
    """Create a sample Workspace object for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir)
        ws = Workspace(workspace_path)
        yield ws


@pytest.fixture
def sample_run_config():
    """Create a sample RunConfig for testing."""
    return RunConfig(
        timeout_sec=60,
        allow_network=False,
        env={"TEST_ENV": "test_value"}
    )


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing."""
    with patch('anthropic.Anthropic') as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    with patch('openai.OpenAI') as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_pydantic_agent():
    """Mock PydanticAI agent for testing."""
    with patch('pydantic_ai.Agent') as mock_agent_class:
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent

        # Mock the run_sync method to return a successful result
        mock_result = MagicMock()
        mock_result.data = "Test agent response"
        mock_result.cost.total_tokens = 100
        mock_result.cost.input_tokens = 60
        mock_result.cost.output_tokens = 40
        mock_agent.run_sync.return_value = mock_result

        yield mock_agent


@pytest.fixture
def sample_query_data():
    """Sample query data for batch processing tests."""
    return {
        "queries": [
            {
                "id": "test_001",
                "query": "How many records are in the dataset?",
                "category": "basic",
                "expected_type": "count"
            },
            {
                "id": "test_002",
                "query": "What are the top 5 pipeline companies by volume?",
                "category": "analysis",
                "expected_type": "ranking"
            }
        ],
        "config": {
            "model": "anthropic:claude-sonnet-4-20250514",
            "max_tools": 15,
            "timeout": 300,
            "template": "templates/data_analysis_agent_prompt.txt"
        }
    }


@pytest.fixture
def sample_config_yaml():
    """Sample configuration YAML for testing."""
    return {
        "dataset": {
            "description": "Natural gas pipeline transportation data",
            "location": "data/sample_data.csv"
        },
        "agent": {
            "model": "anthropic:claude-sonnet-4-20250514",
            "max_tools": 15,
            "timeout": 300,
            "streaming": True
        },
        "analysis": {
            "types": ["exploration", "statistical", "predictive"],
            "rubric": {
                "criteria": ["accuracy", "completeness", "insight"],
                "max_score": 10
            }
        }
    }


@pytest.fixture
def mock_subprocess():
    """Mock subprocess calls for testing."""
    with patch('subprocess.run') as mock_run:
        # Default successful execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Test output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        yield mock_run


@pytest.fixture
def mock_file_operations():
    """Mock file system operations for testing."""
    with patch('pathlib.Path.exists') as mock_exists, \
         patch('pathlib.Path.mkdir') as mock_mkdir, \
         patch('builtins.open', create=True) as mock_open:

        mock_exists.return_value = True
        yield {
            'exists': mock_exists,
            'mkdir': mock_mkdir,
            'open': mock_open
        }


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing write_file_and_run_python."""
    return '''
import pandas as pd
import numpy as np

# Load and analyze data
data = pd.DataFrame({
    'pipeline': ['A', 'B', 'C'],
    'volume': [100, 200, 150]
})

print(f"Total records: {len(data)}")
print(f"Top pipeline: {data.loc[data['volume'].idxmax(), 'pipeline']}")

# Save results
data.to_csv('results.csv', index=False)
'''


@pytest.fixture
def sample_tool_timing():
    """Sample ToolTiming object for testing."""
    return ToolTiming(
        tool_name="write_file_and_run_python",
        start_time=1234567890.0,
        end_time=1234567892.5,
        duration=2.5,
        success=True,
        error=None
    )


@pytest.fixture
def mock_tiktoken():
    """Mock tiktoken for token counting tests."""
    with patch('tiktoken.encoding_for_model') as mock_encoding:
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_encoding.return_value = mock_enc
        yield mock_enc


@pytest.fixture
def sample_pricing_data():
    """Sample pricing data for cost calculation tests."""
    return {
        "claude-sonnet-4-20250514": {
            "input_per_1m": 15.0,
            "output_per_1m": 75.0
        },
        "gpt-4o-mini-2024-07-18": {
            "input_per_1m": 0.15,
            "output_per_1m": 0.60
        }
    }


@pytest.fixture
def mock_environment():
    """Mock environment variables for testing."""
    test_env = {
        'ANTHROPIC_API_KEY': 'test_anthropic_key',
        'OPENAI_API_KEY': 'test_openai_key',
        'TEST_MODE': 'true'
    }
    with patch.dict(os.environ, test_env):
        yield test_env


@pytest.fixture
def sample_react_log():
    """Sample ReAct log entries for testing."""
    return [
        {
            "timestamp": "2025-09-27T13:00:00Z",
            "event_type": "tool_call_start",
            "tool_name": "write_file_and_run_python",
            "args": {"file_path": "analysis.py", "content_length": 150}
        },
        {
            "timestamp": "2025-09-27T13:00:02Z",
            "event_type": "tool_call_end",
            "tool_name": "write_file_and_run_python",
            "success": True,
            "duration": 2.0
        }
    ]


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically set up test environment for all tests."""
    # Ensure we're using the virtual environment python
    original_executable = sys.executable
    test_executable = "/Users/user/Projects/claude_data_agent/.venv/bin/python"

    with patch('sys.executable', test_executable):
        yield

    # Restore original executable
    sys.executable = original_executable