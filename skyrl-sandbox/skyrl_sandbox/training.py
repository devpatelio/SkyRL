"""Background training run manager for the sandbox."""

from __future__ import annotations

import json
import shlex
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, IO, List, Optional

from skyrl_sandbox.models import TrainingLaunchRequest, TrainingLogsResponse, TrainingRunInfo


@dataclass
class TrainingRun:
    """Internal representation of a training run."""

    id: str
    env_name: str
    run_name: str
    command: List[str]
    log_path: Path
    notes: Optional[str] = None
    status: str = "pending"
    process: Optional[subprocess.Popen] = None  # type: ignore[attr-defined]
    log_handle: Optional[IO[str]] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    def as_model(self) -> TrainingRunInfo:
        return TrainingRunInfo(
            id=self.id,
            env_name=self.env_name,
            run_name=self.run_name,
            status=self.status,  # type: ignore[arg-type]
            command=self.command,
            log_path=str(self.log_path),
            started_at=self.started_at,
            finished_at=self.finished_at,
            notes=self.notes,
        )


class TrainingManager:
    """Launch and monitor training processes."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.train_project_root = repo_root / "skyrl-train"
        if not self.train_project_root.exists():
            raise RuntimeError(
                f"Could not find 'skyrl-train' project under {repo_root}. "
                "Training tab requires the full SkyRL repo."
            )

        self.log_dir = Path(__file__).parent / "training_runs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._runs: Dict[str, TrainingRun] = {}
        self._lock = threading.Lock()

    def start_run(self, request: TrainingLaunchRequest) -> TrainingRun:
        if not request.train_data or not request.val_data:
            raise ValueError("Both train_data and val_data must contain at least one path")

        run_id = uuid.uuid4().hex[:8]
        slug = self._slugify(request.run_name or run_id)
        log_path = self.log_dir / f"{slug}-{run_id}.log"
        shell_cmd, command_list = self._build_command(request)

        log_handle = open(log_path, "w", encoding="utf-8")

        # Use bash explicitly when we need to source a venv, otherwise use shell=True with default sh
        if request.venv_path:
            # Use bash explicitly for 'source' command support
            process = subprocess.Popen(
                command_list,  # Already formatted as ["bash", "-c", shell_cmd]
                cwd=str(self.train_project_root),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        else:
            # No venv activation needed, can use shell=True
            process = subprocess.Popen(
                shell_cmd,
                cwd=str(self.train_project_root),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                shell=True,
            )

        run = TrainingRun(
            id=run_id,
            env_name=request.env_name,
            run_name=request.run_name,
            command=command_list,
            log_path=log_path,
            notes=request.notes,
            status="running",
            process=process,
            log_handle=log_handle,
            started_at=datetime.utcnow(),
        )

        with self._lock:
            self._runs[run_id] = run

        watcher = threading.Thread(
            target=self._watch_process,
            args=(run_id,),
            daemon=True,
        )
        watcher.start()

        return run

    def list_runs(self) -> List[TrainingRunInfo]:
        with self._lock:
            return [run.as_model() for run in self._runs.values()]

    def get_run(self, run_id: str) -> TrainingRunInfo:
        with self._lock:
            if run_id not in self._runs:
                raise KeyError(f"Run '{run_id}' not found")
            return self._runs[run_id].as_model()

    def stop_run(self, run_id: str) -> TrainingRunInfo:
        with self._lock:
            if run_id not in self._runs:
                raise KeyError(f"Run '{run_id}' not found")
            run = self._runs[run_id]

        if run.process and run.status == "running":
            run.process.terminate()
            run.status = "stopped"
            run.finished_at = datetime.utcnow()

        if run.log_handle:
            run.log_handle.flush()

        return run.as_model()

    def get_logs(self, run_id: str, tail: int = 200) -> TrainingLogsResponse:
        with self._lock:
            if run_id not in self._runs:
                raise KeyError(f"Run '{run_id}' not found")
            log_path = self._runs[run_id].log_path

        if not log_path.exists():
            raise FileNotFoundError(f"No logs found for run '{run_id}'")

        lines = log_path.read_text(encoding="utf-8").splitlines()
        tail_lines = lines[-tail:] if tail > 0 else lines
        return TrainingLogsResponse(run_id=run_id, tail=tail, lines=tail_lines)

    def _watch_process(self, run_id: str) -> None:
        with self._lock:
            run = self._runs.get(run_id)
        if not run or not run.process:
            return

        return_code = run.process.wait()
        run.finished_at = datetime.utcnow()
        if run.log_handle:
            run.log_handle.flush()
            run.log_handle.close()
            run.log_handle = None

        if run.status != "stopped":
            run.status = "success" if return_code == 0 else "error"

    def _build_command(self, request: TrainingLaunchRequest) -> tuple[str, List[str]]:
        """Build a shell command string for training execution and list representation for storage.
        
        Returns:
            Tuple of (shell_command_string, command_list_for_storage)
        
        The command will:
        1. Activate the virtual environment if venv_path is provided
        2. Run uv with --isolated --extra vllm (or python) to execute training
        3. Include all hydra overrides
        """
        train_paths = [self._resolve_path(path) for path in request.train_data]
        val_paths = [self._resolve_path(path) for path in request.val_data]

        overrides = [
            f"environment.env_class={request.env_name}",
            f"data.train_data={json.dumps(train_paths)}",
            f"data.val_data={json.dumps(val_paths)}",
        ]

        for key, value in request.environment_overrides.items():
            overrides.append(f"{key}={value}")

        overrides.extend(request.hydra_overrides or [])

        # Build the main command with proper escaping
        override_str = " ".join(shlex.quote(ov) for ov in overrides)
        entrypoint_module = self._get_entrypoint_module(request.env_name)
        
        if request.use_uv:
            main_cmd = f"uv run --isolated --extra vllm -m {entrypoint_module} {override_str}"
        else:
            main_cmd = f"python -m {entrypoint_module} {override_str}"
        
        # Build the full command with venv activation
        venv_path = request.venv_path
        shell_cmd = main_cmd
        if venv_path:
            # Expand ~ to full path and escape properly
            venv_path_expanded = str(Path(venv_path).expanduser())
            shell_cmd = f"source {shlex.quote(venv_path_expanded)} && {main_cmd}"
            # For storage, create a readable list representation showing bash execution
            command_list = ["bash", "-c", shell_cmd]
        else:
            # No venv needed, store command as-is
            command_list = shell_cmd.split() if isinstance(shell_cmd, str) else shell_cmd
        
        return shell_cmd, command_list

    def _slugify(self, value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value.lower()).strip("-_") or "run"

    def _resolve_path(self, value: str) -> str:
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Dataset path '{value}' does not exist")
        return str(path.resolve())

    def _get_entrypoint_module(self, env_name: str) -> str:
        """Return a training entrypoint module for the env if a custom script exists.
        
        Priority order:
        1. Check runs/{env_name}/register_env.py (for user-created environments)
        2. Check examples/{env_name}/main_{env_name}.py (for legacy examples)
        3. Fall back to generic skyrl_train.entrypoints.main_base
        """
        # First, check if there's a register_env.py in the runs directory
        runs_module_path = (
            self.train_project_root / "runs" / env_name / "register_env.py"
        )
        if runs_module_path.is_file():
            return f"runs.{env_name}.register_env"
        
        # Second, check examples directory for legacy support
        examples_module_path = (
            self.train_project_root / "examples" / env_name / f"main_{env_name}.py"
        )
        if examples_module_path.is_file():
            return f"examples.{env_name}.main_{env_name}"
        
        # Finally, fall back to the generic entrypoint
        return "skyrl_train.entrypoints.main_base"

