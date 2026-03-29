from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import queue
import subprocess
import threading
from typing import Any
from uuid import uuid4

from speaches.runtime import RuntimeBackendConfig

logger = logging.getLogger(__name__)


class IsolatedPythonWorkerError(RuntimeError):
    pass


_DEBUGGER_ENV_PREFIXES = (
    "PYDEVD_",
    "DEBUGPY_",
    "VSCODE_DEBUGPY_",
)
_DEBUGGER_ENV_KEYS = {
    "PYTHONBREAKPOINT",
}


def _build_worker_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _DEBUGGER_ENV_KEYS and not any(key.startswith(prefix) for prefix in _DEBUGGER_ENV_PREFIXES)
    }
    if extra_env:
        env.update(extra_env)
    return env


class IsolatedPythonProcessClient:
    def __init__(
        self,
        runtime_config: RuntimeBackendConfig,
        *,
        module: str,
        module_args: list[str] | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        if runtime_config.mode != "isolated_python":
            msg = f"Expected isolated_python runtime mode, got '{runtime_config.mode}'"
            raise ValueError(msg)
        if runtime_config.python_executable is None:
            msg = "isolated_python runtime requires python_executable"
            raise ValueError(msg)

        self.runtime_config = runtime_config
        self.module = module
        self.module_args = module_args or []
        self.extra_env = extra_env or {}

        self._process: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[str | None] = queue.Queue()
        self._stderr_queue: queue.Queue[str | None] = queue.Queue()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._request_lock = threading.Lock()

    def start(self) -> dict[str, Any]:
        if self._process is not None:
            msg = "Isolated Python worker is already running"
            raise RuntimeError(msg)

        env = _build_worker_env(self.extra_env)

        src_path = str(Path(__file__).resolve().parents[2])
        python_path_parts = [src_path]
        existing_python_path = env.get("PYTHONPATH")
        if existing_python_path:
            python_path_parts.append(existing_python_path)
        env["PYTHONPATH"] = os.pathsep.join(python_path_parts)

        command = [self.runtime_config.python_executable, "-m", self.module, *self.module_args]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=self.runtime_config.working_directory,
            env=env,
        )
        self._stdout_thread = threading.Thread(
            target=self._stream_pipe,
            args=(self._process.stdout, self._stdout_queue),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._stream_pipe,
            args=(self._process.stderr, self._stderr_queue),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

        ready_message = self._read_message(timeout=self.runtime_config.startup_timeout_seconds)
        if ready_message.get("event") == "ready":
            return ready_message
        error_message = ready_message.get("error", "Worker failed to start")
        self.close()
        raise IsolatedPythonWorkerError(str(error_message))

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._process is None or self._process.stdin is None:
            msg = "Isolated Python worker is not running"
            raise RuntimeError(msg)

        request_id = str(uuid4())
        payload = {
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        with self._request_lock:
            try:
                self._process.stdin.write(json.dumps(payload) + "\n")
                self._process.stdin.flush()
            except BrokenPipeError as error:
                raise IsolatedPythonWorkerError("Worker process terminated unexpectedly") from error

            response = self._read_message(timeout=self.runtime_config.startup_timeout_seconds)
            if response.get("id") != request_id:
                msg = f"Worker response ID mismatch. Expected '{request_id}', got '{response.get('id')}'"
                raise IsolatedPythonWorkerError(msg)
            if response.get("ok") is False:
                raise IsolatedPythonWorkerError(str(response.get("error", "Worker request failed")))
            return response

    def close(self) -> None:
        process = self._process
        if process is None:
            return

        try:
            if process.stdin is not None and process.poll() is None:
                try:
                    process.stdin.write(json.dumps({"id": str(uuid4()), "method": "shutdown", "params": {}}) + "\n")
                    process.stdin.flush()
                except BrokenPipeError:
                    pass
        finally:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            self._process = None

    def _read_message(self, timeout: float) -> dict[str, Any]:
        while True:
            if self._process is not None and self._process.poll() is not None and self._stdout_queue.empty():
                stderr_output = self._collect_stderr()
                msg = "Worker process exited unexpectedly"
                if stderr_output:
                    msg += f": {stderr_output}"
                raise IsolatedPythonWorkerError(msg)

            try:
                line = self._stdout_queue.get(timeout=timeout)
            except queue.Empty as error:
                stderr_output = self._collect_stderr()
                msg = (
                    "Timed out waiting for worker response. "
                    f"Consider increasing startup_timeout_seconds for backend '{self.module}'."
                )
                if stderr_output:
                    msg += f": {stderr_output}"
                raise IsolatedPythonWorkerError(msg) from error

            if line is None:
                stderr_output = self._collect_stderr()
                msg = "Worker closed stdout unexpectedly"
                if stderr_output:
                    msg += f": {stderr_output}"
                raise IsolatedPythonWorkerError(msg)

            try:
                return json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Ignoring non-JSON worker stdout line: %s", line.rstrip())

    @staticmethod
    def _stream_pipe(pipe: Any, target_queue: queue.Queue[str | None]) -> None:
        if pipe is None:
            target_queue.put(None)
            return
        try:
            for line in pipe:
                target_queue.put(line)
        finally:
            target_queue.put(None)

    def _collect_stderr(self) -> str:
        lines: list[str] = []
        while True:
            try:
                line = self._stderr_queue.get_nowait()
            except queue.Empty:
                break
            if line is None:
                continue
            lines.append(line.rstrip())
        return "\n".join(lines)
