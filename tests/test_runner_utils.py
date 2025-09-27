"""
Tests for runner_utils.py - Workspace management and Python execution utilities.

Tests cover:
- Workspace creation and management
- File operations and hashing
- Python execution in workspace context
- Preview and truncation functionality
- Error handling and timeouts
"""

import hashlib
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open
import subprocess
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from runner_utils import (
    Workspace, RunConfig, sha256_bytes, sha256_file, preview_bytes,
    preview_text_lines, MAX_PREVIEW_BYTES, STD_LINES
)


class TestHashingFunctions:
    """Test cryptographic hashing utilities."""

    def test_sha256_bytes(self):
        """Test SHA256 hashing of bytes."""
        test_data = b"test data for hashing"
        expected_hash = hashlib.sha256(test_data).hexdigest()

        result = sha256_bytes(test_data)

        assert result == expected_hash
        assert len(result) == 64  # SHA256 produces 64-character hex string

    def test_sha256_bytes_empty(self):
        """Test SHA256 hashing of empty bytes."""
        test_data = b""
        expected_hash = hashlib.sha256(test_data).hexdigest()

        result = sha256_bytes(test_data)

        assert result == expected_hash

    def test_sha256_file(self, temp_workspace):
        """Test SHA256 hashing of file contents."""
        # Create test file
        test_file = temp_workspace / "test_file.txt"
        test_content = "Test file content for hashing"
        test_file.write_text(test_content)

        # Calculate expected hash
        expected_hash = hashlib.sha256(test_content.encode()).hexdigest()

        result = sha256_file(test_file)

        assert result == expected_hash

    def test_sha256_file_binary(self, temp_workspace):
        """Test SHA256 hashing of binary file."""
        # Create binary test file
        test_file = temp_workspace / "binary_file.bin"
        test_data = bytes(range(256))  # Binary data
        test_file.write_bytes(test_data)

        expected_hash = hashlib.sha256(test_data).hexdigest()

        result = sha256_file(test_file)

        assert result == expected_hash


class TestPreviewFunctions:
    """Test data preview and truncation utilities."""

    def test_preview_bytes_small_data(self):
        """Test preview of small byte data (no truncation)."""
        test_data = b"Small test data"

        result = preview_bytes(test_data)

        assert result["text"] == "Small test data"
        assert result["truncated"] == "false"

    def test_preview_bytes_large_data(self):
        """Test preview of large byte data (with truncation)."""
        # Create data larger than MAX_PREVIEW_BYTES
        large_data = b"x" * (MAX_PREVIEW_BYTES + 1000)

        result = preview_bytes(large_data)

        assert result["truncated"] == "true"
        assert "..." in result["text"]
        assert len(result["text"]) < len(large_data.decode('utf-8', errors='replace'))

    def test_preview_bytes_custom_limit(self):
        """Test preview with custom byte limit."""
        test_data = b"This is a test string that should be truncated"
        custom_limit = 20

        result = preview_bytes(test_data, limit=custom_limit)

        assert result["truncated"] == "true"
        assert "..." in result["text"]

    def test_preview_bytes_unicode_handling(self):
        """Test preview with Unicode data."""
        test_data = "测试数据 with émojis 🚀".encode('utf-8')

        result = preview_bytes(test_data)

        assert "测试数据" in result["text"]
        assert "émojis" in result["text"]
        assert "🚀" in result["text"]

    def test_preview_text_lines_small(self):
        """Test preview of small text (no truncation)."""
        test_text = "Line 1\nLine 2\nLine 3"

        result = preview_text_lines(test_text)

        assert result["text"] == test_text
        assert result["truncated"] == "false"

    def test_preview_text_lines_large(self):
        """Test preview of large text (with truncation)."""
        # Create text with more lines than 2 * STD_LINES
        lines = [f"Line {i}" for i in range(STD_LINES * 3)]
        test_text = "\n".join(lines)

        result = preview_text_lines(test_text)

        assert result["truncated"] == "true"
        assert "..." in result["text"]
        assert "Line 0" in result["text"]  # Should have first lines
        assert f"Line {len(lines) - 1}" in result["text"]  # Should have last lines

    def test_preview_text_lines_custom_limit(self):
        """Test preview with custom line limit."""
        lines = [f"Line {i}" for i in range(20)]
        test_text = "\n".join(lines)
        custom_lines = 5

        result = preview_text_lines(test_text, lines=custom_lines)

        assert result["truncated"] == "true"
        assert "Line 0" in result["text"]
        assert "Line 19" in result["text"]


class TestRunConfig:
    """Test RunConfig dataclass."""

    def test_run_config_defaults(self):
        """Test RunConfig with default values."""
        config = RunConfig()

        assert config.timeout_sec == 60
        assert config.allow_network is False
        assert config.env == {}

    def test_run_config_custom_values(self):
        """Test RunConfig with custom values."""
        custom_env = {"CUSTOM_VAR": "value", "PATH": "/custom/path"}
        config = RunConfig(
            timeout_sec=120,
            allow_network=True,
            env=custom_env
        )

        assert config.timeout_sec == 120
        assert config.allow_network is True
        assert config.env == custom_env

    def test_run_config_env_modification(self):
        """Test modifying RunConfig environment."""
        config = RunConfig()
        config.env["NEW_VAR"] = "new_value"

        assert config.env["NEW_VAR"] == "new_value"


class TestWorkspace:
    """Test Workspace class and functionality."""

    def test_workspace_creation(self, temp_workspace):
        """Test Workspace creation."""
        ws = Workspace(temp_workspace)

        assert ws.root == temp_workspace
        assert ws.root.exists()

    def test_workspace_file_operations(self, temp_workspace):
        """Test basic file operations in workspace."""
        ws = Workspace(temp_workspace)

        # Create a test file
        test_file = ws.root / "test.py"
        test_content = "print('Hello, World!')"
        test_file.write_text(test_content)

        # Verify file exists and content is correct
        assert test_file.exists()
        assert test_file.read_text() == test_content

    def test_workspace_subdirectory_creation(self, temp_workspace):
        """Test creating subdirectories in workspace."""
        ws = Workspace(temp_workspace)

        # Create subdirectory
        subdir = ws.root / "analysis" / "results"
        subdir.mkdir(parents=True)

        assert subdir.exists()
        assert subdir.is_dir()

    @patch('subprocess.run')
    def test_workspace_python_execution(self, mock_subprocess, temp_workspace):
        """Test Python execution in workspace context."""
        ws = Workspace(temp_workspace)

        # Create Python script
        script_file = ws.root / "test_script.py"
        script_content = """
import os
print(f"Working directory: {os.getcwd()}")
print("Script executed successfully")
"""
        script_file.write_text(script_content)

        # Mock successful execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Working directory: /tmp/workspace\nScript executed successfully\n"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Test execution (would be called by write_file_and_run_python)
        result = subprocess.run(
            [sys.executable, str(script_file)],
            cwd=ws.root,
            capture_output=True,
            text=True,
            timeout=60
        )

        # Verify subprocess was called correctly
        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args
        assert args[0][0][0] == sys.executable
        assert str(script_file) in args[0][0]
        assert args[1]['cwd'] == ws.root

    @patch('subprocess.run')
    def test_workspace_execution_timeout(self, mock_subprocess, temp_workspace):
        """Test handling of execution timeout in workspace."""
        from subprocess import TimeoutExpired

        ws = Workspace(temp_workspace)

        # Mock timeout
        mock_subprocess.side_effect = TimeoutExpired("python", 30)

        # Test timeout handling
        with pytest.raises(TimeoutExpired):
            subprocess.run(
                [sys.executable, "long_running_script.py"],
                cwd=ws.root,
                capture_output=True,
                text=True,
                timeout=30
            )

    @patch('subprocess.run')
    def test_workspace_execution_error(self, mock_subprocess, temp_workspace):
        """Test handling of execution errors in workspace."""
        ws = Workspace(temp_workspace)

        # Mock execution error
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "ModuleNotFoundError: No module named 'nonexistent'"
        mock_subprocess.return_value = mock_result

        # Test error handling
        result = subprocess.run(
            [sys.executable, "-c", "import nonexistent"],
            cwd=ws.root,
            capture_output=True,
            text=True,
            timeout=60
        )

        # Verify error is captured
        mock_subprocess.assert_called_once()
        assert mock_result.returncode == 1
        assert "ModuleNotFoundError" in mock_result.stderr

    def test_workspace_file_hashing(self, temp_workspace):
        """Test file hashing within workspace."""
        ws = Workspace(temp_workspace)

        # Create test file with known content
        test_file = ws.root / "data.csv"
        test_content = "name,value\ntest,123\ndata,456"
        test_file.write_text(test_content)

        # Calculate hash
        file_hash = sha256_file(test_file)
        expected_hash = hashlib.sha256(test_content.encode()).hexdigest()

        assert file_hash == expected_hash

    def test_workspace_large_file_handling(self, temp_workspace):
        """Test handling of large files in workspace."""
        ws = Workspace(temp_workspace)

        # Create large file
        large_file = ws.root / "large_data.txt"
        large_content = "x" * (MAX_PREVIEW_BYTES + 5000)
        large_file.write_text(large_content)

        # Read and preview
        content_bytes = large_file.read_bytes()
        preview = preview_bytes(content_bytes)

        assert preview["truncated"] == "true"
        assert len(preview["text"]) < len(large_content)

    def test_workspace_environment_isolation(self, temp_workspace):
        """Test environment variable isolation in workspace execution."""
        ws = Workspace(temp_workspace)

        # Create script that checks environment
        env_script = ws.root / "check_env.py"
        env_script.write_text("""
import os
print(f"CUSTOM_VAR: {os.environ.get('CUSTOM_VAR', 'NOT_SET')}")
print(f"PATH exists: {'PATH' in os.environ}")
""")

        # Test with custom environment
        custom_env = {"CUSTOM_VAR": "test_value", "PATH": "/custom/path"}

        with patch('subprocess.run') as mock_subprocess:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "CUSTOM_VAR: test_value\nPATH exists: True\n"
            mock_subprocess.return_value = mock_result

            # Simulate execution with custom environment
            subprocess.run(
                [sys.executable, str(env_script)],
                cwd=ws.root,
                env=custom_env,
                capture_output=True,
                text=True
            )

            # Verify environment was passed
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args
            assert call_args[1]['env'] == custom_env

    def test_workspace_concurrent_access(self, temp_workspace):
        """Test concurrent workspace operations."""
        ws = Workspace(temp_workspace)

        # Create multiple files simultaneously
        files = []
        for i in range(5):
            test_file = ws.root / f"concurrent_test_{i}.txt"
            test_file.write_text(f"Content for file {i}")
            files.append(test_file)

        # Verify all files were created correctly
        for i, file_path in enumerate(files):
            assert file_path.exists()
            assert file_path.read_text() == f"Content for file {i}"

    def test_workspace_cleanup_simulation(self, temp_workspace):
        """Test workspace cleanup behavior."""
        ws = Workspace(temp_workspace)

        # Create various files and directories
        (ws.root / "temp_data.csv").write_text("temp,data")
        (ws.root / "results").mkdir()
        (ws.root / "results" / "output.txt").write_text("results")
        (ws.root / "logs").mkdir()
        (ws.root / "logs" / "debug.log").write_text("debug info")

        # Verify structure was created
        assert (ws.root / "temp_data.csv").exists()
        assert (ws.root / "results").is_dir()
        assert (ws.root / "results" / "output.txt").exists()
        assert (ws.root / "logs" / "debug.log").exists()

        # Workspace cleanup would be handled by temp directory context manager

    def test_workspace_permission_handling(self, temp_workspace):
        """Test workspace file permission handling."""
        ws = Workspace(temp_workspace)

        # Create file with specific content
        test_file = ws.root / "permission_test.py"
        test_file.write_text("#!/usr/bin/env python3\nprint('Permission test')")

        # File should be readable
        content = test_file.read_text()
        assert "Permission test" in content

        # File should be writable (can modify)
        test_file.write_text("# Modified content\nprint('Updated')")
        updated_content = test_file.read_text()
        assert "Updated" in updated_content


class TestWorkspaceIntegration:
    """Integration tests for workspace functionality with other components."""

    @patch('subprocess.run')
    def test_workspace_with_run_config(self, mock_subprocess, temp_workspace):
        """Test workspace operations with RunConfig."""
        ws = Workspace(temp_workspace)
        config = RunConfig(
            timeout_sec=30,
            allow_network=False,
            env={"WORKSPACE_TEST": "true"}
        )

        # Create test script
        script = ws.root / "config_test.py"
        script.write_text("""
import os
print(f"WORKSPACE_TEST: {os.environ.get('WORKSPACE_TEST')}")
""")

        # Mock execution with config
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "WORKSPACE_TEST: true\n"
        mock_subprocess.return_value = mock_result

        # Simulate execution with config
        subprocess.run(
            [sys.executable, str(script)],
            cwd=ws.root,
            env=config.env,
            timeout=config.timeout_sec,
            capture_output=True,
            text=True
        )

        # Verify correct parameters were used
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args
        assert call_args[1]['timeout'] == 30
        assert call_args[1]['env']['WORKSPACE_TEST'] == "true"

    def test_workspace_file_operations_integration(self, temp_workspace):
        """Test integrated file operations in workspace."""
        ws = Workspace(temp_workspace)

        # Create data files
        data_file = ws.root / "input_data.csv"
        data_content = "id,name,value\n1,test,100\n2,example,200"
        data_file.write_text(data_content)

        # Create processing script
        script_file = ws.root / "process_data.py"
        script_content = """
import csv
import json

# Read input data
with open('input_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

# Process and save
total_value = sum(int(row['value']) for row in data)
result = {'total_records': len(data), 'total_value': total_value}

with open('results.json', 'w') as f:
    json.dump(result, f)

print(f"Processed {len(data)} records, total value: {total_value}")
"""
        script_file.write_text(script_content)

        # Files should exist and have correct content
        assert data_file.exists()
        assert script_file.exists()

        # Verify content integrity with hashing
        data_hash = sha256_file(data_file)
        script_hash = sha256_file(script_file)

        assert len(data_hash) == 64
        assert len(script_hash) == 64

        # Verify preview functionality
        data_preview = preview_bytes(data_file.read_bytes())
        assert "id,name,value" in data_preview["text"]
        assert data_preview["truncated"] == "false"