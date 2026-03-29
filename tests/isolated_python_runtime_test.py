import sys

import pytest

from speaches.executors.shared.base_model_manager import BaseModelManager
from speaches.runtime import RuntimeBackendConfig
from speaches.runtime_backends import IsolatedPythonProcessClient
from speaches.runtime_backends.isolated_python import _build_worker_env


def test_isolated_python_process_client_round_trip() -> None:
    client = IsolatedPythonProcessClient(
        RuntimeBackendConfig(mode="isolated_python", python_executable=sys.executable, startup_timeout_seconds=5),
        module="tests.fake_isolated_worker",
    )

    ready = client.start()
    assert ready["event"] == "ready"
    assert ready["voices"] == ["speaker_a"]

    response = client.request("ping")
    assert response["ok"] is True
    assert response["result"] == {"pong": True}

    client.close()


def test_build_worker_env_strips_debugger_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYDEVD_DISABLE_FILE_VALIDATION", "1")
    monkeypatch.setenv("DEBUGPY_RUNNING", "1")
    monkeypatch.setenv("VSCODE_DEBUGPY_ADAPTER_ENDPOINTS", "/tmp/debugpy")
    monkeypatch.setenv("PYTHONBREAKPOINT", "debugpy.breakpoint")
    monkeypatch.setenv("KEEP_ME", "1")

    env = _build_worker_env({"EXTRA_KEY": "value"})

    assert "PYDEVD_DISABLE_FILE_VALIDATION" not in env
    assert "DEBUGPY_RUNNING" not in env
    assert "VSCODE_DEBUGPY_ADAPTER_ENDPOINTS" not in env
    assert "PYTHONBREAKPOINT" not in env
    assert env["KEEP_ME"] == "1"
    assert env["EXTRA_KEY"] == "value"


class _ClosableModel:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _DummyManager(BaseModelManager[_ClosableModel]):
    def _load_fn(self, model_id: str) -> _ClosableModel:
        return _ClosableModel()

    def _unload_fn(self, model: _ClosableModel) -> None:
        model.close()


def test_base_model_manager_calls_unload_hook() -> None:
    manager = _DummyManager(ttl=-1)
    handle = manager.load_model("dummy-model")

    with handle as model:
        assert model.closed is False

    loaded_model = handle.model
    assert loaded_model is not None
    manager.unload_model("dummy-model")
    assert loaded_model.closed is True
