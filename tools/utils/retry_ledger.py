"""
Retry Ledger System for PydanticAI Tool Call Investigation

This module provides surgical instrumentation to identify exactly where retries are being burned:
1. Pre-dispatch: Bad tool calls, validation failures
2. Execution: Real script errors, exit codes
3. Post-process: Schema/serialization issues

Based on colleague's advice to distinguish between retry causes.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

class RetryPhase(Enum):
    PRE_DISPATCH = "pre_dispatch"
    EXECUTION = "execution"
    POST_PROCESS = "post_process"

@dataclass
class RetryAttempt:
    attempt_number: int
    timestamp: str
    phase: RetryPhase
    tool_name: str
    retry_delta: int  # +1 if this consumed retry budget, +0 if not

    # Phase-specific data
    raw_args: Optional[Dict[str, Any]] = None
    validation_error: Optional[str] = None
    exit_code: Optional[int] = None
    duration_s: Optional[float] = None
    stderr_path: Optional[str] = None
    stdout_tail: Optional[str] = None
    stderr_tail: Optional[str] = None
    schema_error: Optional[str] = None

    def to_jsonl(self) -> str:
        """Convert to JSONL format for logging"""
        data = asdict(self)
        data['phase'] = self.phase.value
        return json.dumps(data, separators=(',', ':'))

class RetryLedger:
    """Tracks all retry attempts across the three phases"""

    def __init__(self, query_id: str, workspace_dir: Path):
        self.query_id = query_id
        self.workspace_dir = workspace_dir
        self.attempts: List[RetryAttempt] = []
        self.current_attempt = 0
        self.ledger_file = workspace_dir / "retry_ledger.jsonl"

        # Ensure workspace exists
        workspace_dir.mkdir(parents=True, exist_ok=True)

    def log_attempt(self, attempt: RetryAttempt) -> None:
        """Log a retry attempt to both memory and disk"""
        self.attempts.append(attempt)

        # Write immediately to disk for real-time tracking
        with open(self.ledger_file, 'a') as f:
            f.write(attempt.to_jsonl() + '\n')

    def log_pre_dispatch(self, tool_name: str, raw_args: Dict[str, Any],
                        validation_error: Optional[str] = None) -> None:
        """Log pre-dispatch validation failure"""
        self.current_attempt += 1
        retry_delta = 1 if validation_error else 0

        attempt = RetryAttempt(
            attempt_number=self.current_attempt,
            timestamp=datetime.now().isoformat(),
            phase=RetryPhase.PRE_DISPATCH,
            tool_name=tool_name,
            retry_delta=retry_delta,
            raw_args=raw_args,
            validation_error=validation_error
        )
        self.log_attempt(attempt)

    def log_execution(self, tool_name: str, exit_code: int, duration_s: float,
                     stderr_path: Optional[str] = None, stdout_tail: Optional[str] = None,
                     stderr_tail: Optional[str] = None) -> None:
        """Log execution phase result"""
        # Only increment attempt if this is a new execution (not already counted in pre-dispatch)
        if not self.attempts or self.attempts[-1].phase != RetryPhase.PRE_DISPATCH:
            self.current_attempt += 1

        retry_delta = 1 if exit_code != 0 else 0

        attempt = RetryAttempt(
            attempt_number=self.current_attempt,
            timestamp=datetime.now().isoformat(),
            phase=RetryPhase.EXECUTION,
            tool_name=tool_name,
            retry_delta=retry_delta,
            exit_code=exit_code,
            duration_s=duration_s,
            stderr_path=stderr_path,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail
        )
        self.log_attempt(attempt)

    def log_post_process(self, tool_name: str, schema_error: Optional[str] = None) -> None:
        """Log post-processing validation failure"""
        if not self.attempts or self.attempts[-1].phase != RetryPhase.EXECUTION:
            self.current_attempt += 1

        retry_delta = 1 if schema_error else 0

        attempt = RetryAttempt(
            attempt_number=self.current_attempt,
            timestamp=datetime.now().isoformat(),
            phase=RetryPhase.POST_PROCESS,
            tool_name=tool_name,
            retry_delta=retry_delta,
            schema_error=schema_error
        )
        self.log_attempt(attempt)

    def print_retry_summary(self, error_message: str = "exceeded max retries") -> None:
        """Print one-screen retry summary as suggested by colleague"""
        print(f"\n{'='*60}")
        print(f"RETRY SUMMARY: {self.query_id}")
        print(f"{'='*60}")

        if not self.attempts:
            print("No retry attempts recorded")
            return

        tool_name = self.attempts[0].tool_name
        print(f"tool: {tool_name}")
        print(f"attempts:")

        for attempt in self.attempts:
            phase_symbol = {
                RetryPhase.PRE_DISPATCH: "🔧",
                RetryPhase.EXECUTION: "⚡",
                RetryPhase.POST_PROCESS: "📤"
            }.get(attempt.phase, "❓")

            retry_indicator = f"retry=+{attempt.retry_delta}"

            if attempt.phase == RetryPhase.PRE_DISPATCH:
                detail = f"invalid_args: {json.dumps(attempt.raw_args, separators=(',', ':'))[:50]}..."
                if attempt.validation_error:
                    detail = f"validation_error: {attempt.validation_error}"

            elif attempt.phase == RetryPhase.EXECUTION:
                detail = f"exit_code={attempt.exit_code}"
                if attempt.stderr_tail:
                    detail += f" error={attempt.stderr_tail[:50]}..."

            elif attempt.phase == RetryPhase.POST_PROCESS:
                detail = f"schema_error: {attempt.schema_error}" if attempt.schema_error else "success"

            print(f"  {attempt.attempt_number}) {phase_symbol} {attempt.phase.value:<12} {detail:<40} {retry_indicator}")

        total_retries = sum(a.retry_delta for a in self.attempts)
        print(f"error: {error_message} (burned {total_retries} retries)")
        print(f"{'='*60}")

    def get_histogram(self) -> Dict[str, int]:
        """Get histogram of retry causes for analysis"""
        histogram = {
            "pre_dispatch_failures": 0,
            "execution_failures": 0,
            "post_process_failures": 0,
            "total_retries_burned": 0
        }

        for attempt in self.attempts:
            if attempt.retry_delta > 0:
                histogram["total_retries_burned"] += attempt.retry_delta

                if attempt.phase == RetryPhase.PRE_DISPATCH:
                    histogram["pre_dispatch_failures"] += 1
                elif attempt.phase == RetryPhase.EXECUTION:
                    histogram["execution_failures"] += 1
                elif attempt.phase == RetryPhase.POST_PROCESS:
                    histogram["post_process_failures"] += 1

        return histogram

# Global ledger instance (will be set by the executor)
_current_ledger: Optional[RetryLedger] = None

def set_current_ledger(ledger: RetryLedger) -> None:
    """Set the current ledger for this query execution"""
    global _current_ledger
    _current_ledger = ledger

def get_current_ledger() -> Optional[RetryLedger]:
    """Get the current ledger instance"""
    return _current_ledger

def log_pre_dispatch(tool_name: str, raw_args: Dict[str, Any],
                    validation_error: Optional[str] = None) -> None:
    """Convenience function to log pre-dispatch events"""
    if _current_ledger:
        _current_ledger.log_pre_dispatch(tool_name, raw_args, validation_error)

def log_execution(tool_name: str, exit_code: int, duration_s: float,
                 stderr_path: Optional[str] = None, stdout_tail: Optional[str] = None,
                 stderr_tail: Optional[str] = None) -> None:
    """Convenience function to log execution events"""
    if _current_ledger:
        _current_ledger.log_execution(tool_name, exit_code, duration_s,
                                    stderr_path, stdout_tail, stderr_tail)

def log_post_process(tool_name: str, schema_error: Optional[str] = None) -> None:
    """Convenience function to log post-process events"""
    if _current_ledger:
        _current_ledger.log_post_process(tool_name, schema_error)

def print_retry_summary(error_message: str = "exceeded max retries") -> None:
    """Convenience function to print retry summary"""
    if _current_ledger:
        _current_ledger.print_retry_summary(error_message)