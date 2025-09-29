"""Adapter to execute ideas through existing Claude Data Agent"""

import json
import subprocess
import tempfile
import uuid
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from ..core.specifications import IdeaSpec


class ExecutionAdapter:
    """Adapter for executing ideas through existing agent system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.template_dir = "src/idea_generation_mvp/templates"
        self.results_dir = Path("results")
        self.system_config = self._load_system_config()

    def spec_to_query(self, spec: IdeaSpec) -> str:
        """Convert IdeaSpec to natural language query"""
        query = spec.to_query_text()

        # Add method-specific guidance
        method_guidance = {
            "event_study": "Use event study methodology with pre-trend testing.",
            "did": "Apply difference-in-differences with proper controls.",
            "synthetic_control": "Use synthetic control method with donor pool validation.",
            "constraint_detection": "Identify capacity constraints using utilization analysis.",
            "lead_lag_analysis": "Analyze lead-lag relationships using VAR or Granger causality.",
            "anomaly_detection": "Detect anomalies using statistical methods.",
            "storage_behavior_detection": "Identify storage-like behavior patterns."
        }

        if spec.method in method_guidance:
            query += f" {method_guidance[spec.method]}"

        # Add trader context
        if spec.trader_hook:
            query += f" Business context: {spec.trader_hook}"

        return query

    def spec_to_batch_format(self, spec: IdeaSpec) -> Dict[str, Any]:
        """Convert IdeaSpec to batch query format"""
        return {
            "id": spec.id,
            "category": self._determine_category(spec),
            "analysis_type": self._determine_analysis_type(spec),
            "query": self.spec_to_query(spec),
            "expected_answer_type": "analysis_results",
            "description": spec.query,
            "critical_warning": "Generated idea - validate results carefully"
        }

    def create_batch_file(self, specs: list[IdeaSpec]) -> str:
        """Create a batch query file for multiple specs"""
        batch_data = {
            "dataset_info": {
                "name": "US Gas Pipeline Transportation Data",
                "file_path": self._get_dataset_path(),
                "record_count": 23854855,
                "date_range": "2022-01-01 to 2025-08-26"
            },
            "queries": [self.spec_to_batch_format(spec) for spec in specs]
        }

        # Create temporary file
        temp_file = Path(f"queries/mvp_batch_{uuid.uuid4().hex[:8]}.json")
        temp_file.parent.mkdir(exist_ok=True)

        with open(temp_file, 'w') as f:
            json.dump(batch_data, f, indent=2)

        return str(temp_file)

    def execute_single(self, spec: IdeaSpec) -> Dict[str, Any]:
        """Execute a single idea"""
        query = self.spec_to_query(spec)

        try:
            # Run through existing agent
            cmd = [
                "python", "run_agent.py",
                "--task", query,
                "--model", self.config.get("model", "anthropic:claude-sonnet-4-20250514"),
                "--max-tools", str(self.config.get("max_tools", 15))
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.get("timeout_per_idea", 300),
                cwd="."
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "spec": spec,
                    "execution_time": None  # Would need to parse from output
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr,
                    "spec": spec,
                    "execution_time": None
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Execution timeout",
                "spec": spec,
                "execution_time": self.config.get("timeout_per_idea", 300)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "spec": spec,
                "execution_time": None
            }

    def execute_batch(self, specs: list[IdeaSpec]) -> Dict[str, Any]:
        """Execute multiple ideas as a batch"""
        batch_file = self.create_batch_file(specs)

        try:
            cmd = [
                "python", "run_batch_queries.py",
                batch_file,
                "--count", str(len(specs)),
                "--workers", str(self.config.get("workers", 1))
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.get("timeout_per_idea", 300) * len(specs),
                cwd="."
            )

            if result.returncode == 0:
                # Find the results file
                results_file = self._find_latest_results_file()
                if results_file:
                    with open(results_file, 'r') as f:
                        batch_results = json.load(f)

                    return {
                        "success": True,
                        "batch_results": batch_results,
                        "results_file": str(results_file),
                        "specs": specs
                    }

            return {
                "success": False,
                "error": result.stderr,
                "specs": specs
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "specs": specs
            }

    def _load_system_config(self) -> Dict[str, Any]:
        """Load system configuration from config.yaml"""
        config_path = Path("config/config.yaml")

        # Try relative to project root from different working directories
        for base_path in [Path("."), Path("../.."), Path("../../..")]:
            full_path = base_path / config_path
            if full_path.exists():
                with open(full_path, 'r') as f:
                    return yaml.safe_load(f)

        # Fallback to empty config if not found
        return {}

    def _get_dataset_path(self) -> str:
        """Get dataset path based on configuration"""
        dataset_config = self.system_config.get("dataset", {})

        # Check if absolute path mode is enabled
        if dataset_config.get("use_absolute_path", False):
            absolute_path = dataset_config.get("absolute_path")
            if absolute_path and Path(absolute_path).exists():
                return absolute_path

        # Fallback to relative path
        return dataset_config.get("path", "data/pipeline_data.parquet")

    def _determine_category(self, spec: IdeaSpec) -> str:
        """Determine query category from spec"""
        if spec.event_context:
            if "lng" in spec.event_context.lower():
                return "lng_analysis"
            elif "weather" in spec.event_context.lower() or "storm" in spec.event_context.lower():
                return "weather_impact"
            elif "expansion" in spec.event_context.lower() or "pipeline" in spec.event_context.lower():
                return "infrastructure"

        if spec.method:
            if "constraint" in spec.method:
                return "capacity_analysis"
            elif "anomaly" in spec.method:
                return "anomaly"
            elif "storage" in spec.method:
                return "storage_analysis"

        return "general_analysis"

    def _determine_analysis_type(self, spec: IdeaSpec) -> str:
        """Determine analysis type from spec"""
        if spec.design_type == "causal":
            return "causal"
        elif "pattern" in spec.method or "storage" in spec.method or "constraint" in spec.method:
            return "pattern"
        elif "anomaly" in spec.method:
            return "anomaly"
        else:
            return "factual"

    def _find_latest_results_file(self) -> Optional[Path]:
        """Find the most recent results file"""
        results_files = list(self.results_dir.glob("batch_results_*.json"))
        if results_files:
            return max(results_files, key=lambda f: f.stat().st_mtime)
        return None

    def extract_artifacts(self, run_dir: str) -> Dict[str, Any]:
        """Extract artifacts from a completed run"""
        run_path = Path(run_dir)
        artifacts = {}

        if run_path.exists():
            # Look for common artifact types
            for pattern in ["*.json", "*.png", "*.csv", "*.txt"]:
                files = list(run_path.glob(pattern))
                for file in files:
                    artifacts[file.name] = str(file)

        return artifacts