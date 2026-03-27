from typing import Literal

from pydantic import BaseModel, Field

type RuntimeMode = Literal["in_process", "isolated_python"]


class RuntimeBackendConfig(BaseModel):
    mode: RuntimeMode = "in_process"
    python_executable: str | None = None
    working_directory: str | None = None
    startup_timeout_seconds: int = Field(default=60, ge=1)


class RuntimeBackendsConfig(BaseModel):
    faster_whisper: RuntimeBackendConfig = RuntimeBackendConfig()
    kokoro: RuntimeBackendConfig = RuntimeBackendConfig()
    onnxruntime: RuntimeBackendConfig = RuntimeBackendConfig()
    vllm: RuntimeBackendConfig = RuntimeBackendConfig(
        mode="isolated_python",
        python_executable=".venv-vllm/bin/python",
    )
    vllm_omni: RuntimeBackendConfig = RuntimeBackendConfig(
        mode="isolated_python",
        python_executable=".venv-vllm/bin/python",
    )

    def get_backend(self, backend_name: str | None) -> RuntimeBackendConfig | None:
        if backend_name is None:
            return None
        return getattr(self, backend_name, None)
