import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any

MAX_PREVIEW_BYTES = 24_000  # trim large file echoes
STD_LINES = 30              # head/tail lines for stdout/stderr previews

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def sha256_file(path: Path) -> str:
    with open(path, 'rb') as f:
        return sha256_bytes(f.read())

def preview_bytes(data: bytes, limit: int = MAX_PREVIEW_BYTES) -> Dict[str, str]:
    if len(data) <= limit:
        return {"text": data.decode('utf-8', errors='replace'), "truncated": "false"}
    head = data[: limit // 2]
    tail = data[-limit // 2 :]
    return {
        "text": head.decode('utf-8', errors='replace') + "\n...\n" + tail.decode('utf-8', errors='replace'),
        "truncated": "true",
    }

def preview_text_lines(text: str, lines: int = STD_LINES) -> Dict[str, str]:
    parts = text.splitlines()
    if len(parts) <= 2 * lines:
        return {"text": text, "truncated": "false"}
    head = "\n".join(parts[:lines])
    tail = "\n".join(parts[-lines:])
    return {"text": head + "\n...\n" + tail, "truncated": "true"}

@dataclass
class RunConfig:
    timeout_sec: int = 60
    allow_network: bool = False
    env: Dict[str, str] = field(default_factory=dict)

@dataclass
class Workspace:
    base_dir: Path
    run_dir: Path = field(init=False)
    workspace: Path = field(init=False)
    manifest: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        ts = time.strftime('%Y%m%d-%H%M%S')
        rid = f"{ts}-{os.urandom(4).hex()}"
        self.run_dir = (self.base_dir / rid)
        self.workspace = (self.run_dir / "workspace")
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)

        # Create symlink to data directory in workspace for easy access
        project_data = Path.cwd() / "data"
        if project_data.exists() and project_data.is_dir():
            data_link = self.workspace / "data"
            if not data_link.exists():
                try:
                    data_link.symlink_to(project_data)
                except OSError:
                    # If symlink fails, continue without it
                    pass

        self.manifest = {"run_id": rid, "files": {}}

    def _safe_path(self, rel: str) -> Path:
        # For absolute paths, use as-is
        if Path(rel).is_absolute():
            return Path(rel).resolve()

        # For relative paths, start from workspace
        return (self.workspace / rel).resolve()

    def write_file(self, path: str, contents: str, executable: bool = False) -> Dict[str, Any]:
        p = self._safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = contents.encode('utf-8')
        with open(p, 'wb') as f:
            f.write(data)
        if executable:
            mode = os.stat(p).st_mode
            os.chmod(p, mode | 0o111)
        info = {
            "path": str(p.relative_to(self.workspace.resolve())),
            "bytes": len(data),
            "sha256": sha256_bytes(data),
        }
        self.manifest["files"][info["path"]] = info
        return {
            **info,
            "preview": preview_bytes(data),
        }

    def read_file(self, path: str) -> Dict[str, Any]:
        p = self._safe_path(path)
        data = p.read_bytes()

        # Try to get relative path, fallback to absolute if outside workspace
        try:
            relative_path = str(p.relative_to(self.workspace.resolve()))
        except ValueError:
            relative_path = str(p)

        return {
            "path": relative_path,
            "bytes": len(data),
            "sha256": sha256_bytes(data),
            "preview": preview_bytes(data),
        }

    def list_files(self) -> List[Dict[str, Any]]:
        out = []
        workspace_resolved = self.workspace.resolve()
        for p in self.workspace.rglob('*'):
            if p.is_file():
                rel = str(p.resolve().relative_to(workspace_resolved))
                out.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p)})
        return out

    def run_python(self, entrypoint: str = "main.py", args: List[str] = None, cfg: RunConfig = None) -> Dict[str, Any]:
        args = args or []
        cfg = cfg or RunConfig()

        ep = self._safe_path(entrypoint)
        if not ep.exists():
            raise FileNotFoundError(f"Entrypoint not found: {entrypoint}")

        cmd = [sys.executable, str(ep), *args]
        env = {k: v for k, v in os.environ.items()}
        env.update(cfg.env or {})
        # Policy knob: indicate to scripts they must avoid network.
        env["NO_NETWORK"] = "1"
        start = time.time()
        proc = subprocess.Popen(
            cmd,
            cwd=str(self.workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        try:
            stdout, stderr = proc.communicate(timeout=cfg.timeout_sec)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            rc = -9

        wall = time.time() - start

        # Save full logs
        logs_dir = (self.run_dir / "logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        ls = int(time.time())
        out_path = logs_dir / f"stdout-{ls}.log"
        err_path = logs_dir / f"stderr-{ls}.log"
        out_path.write_text(stdout, encoding='utf-8', errors='ignore')
        err_path.write_text(stderr, encoding='utf-8', errors='ignore')

        return {
            "cmd": cmd,
            "exit_code": rc,
            "wall_time_sec": round(wall, 3),
            "stdout_preview": preview_text_lines(stdout),
            "stderr_preview": preview_text_lines(stderr),
            "stdout_log": str(out_path),
            "stderr_log": str(err_path),
        }
