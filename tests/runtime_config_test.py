from speaches.config import Config
from speaches.executors.shared.registry import ExecutorRegistry
from speaches.runtime import RuntimeBackendConfig


def test_default_runtime_modes_are_split_by_backend() -> None:
    config = Config(enable_ui=False)

    assert config.runtimes.onnxruntime.mode == "in_process"
    assert config.runtimes.kokoro.mode == "in_process"
    assert config.runtimes.vllm.mode == "isolated_python"
    assert config.runtimes.vllm_omni.mode == "isolated_python"
    assert config.runtimes.vllm.python_executable == ".venv-vllm/bin/python"
    assert config.runtimes.vllm_omni.python_executable == ".venv-vllm/bin/python"


def test_executor_registry_assigns_runtime_config_by_backend() -> None:
    config = Config(enable_ui=False)
    executor_registry = ExecutorRegistry(config)

    assert executor_registry.vad.runtime_backend == "onnxruntime"
    assert executor_registry.vad.runtime_mode == "in_process"
    assert executor_registry.text_to_speech[0].runtime_backend == "onnxruntime"
    assert executor_registry.text_to_speech[1].runtime_backend == "kokoro"
    assert executor_registry.text_to_speech[1].runtime_mode == "in_process"


def test_executor_registry_groups_executors_by_runtime_backend() -> None:
    configured_runtimes = Config(enable_ui=False).runtimes.model_copy(
        update={
            "onnxruntime": RuntimeBackendConfig(
                mode="isolated_python",
                python_executable=".venv-onnx/bin/python",
            )
        }
    )
    config = Config(enable_ui=False, runtimes=configured_runtimes)
    executor_registry = ExecutorRegistry(config)

    assert configured_runtimes.onnxruntime.mode == "isolated_python"
    assert {executor.name for executor in executor_registry.executors_by_runtime_backend()["onnxruntime"]} == {
        "parakeet",
        "piper",
        "wespeaker-speaker-embedding",
        "pyannote-speaker-segmentation",
        "vad",
    }
    assert all(
        executor.runtime_mode == "isolated_python"
        for executor in executor_registry.executors_by_runtime_backend()["onnxruntime"]
    )
