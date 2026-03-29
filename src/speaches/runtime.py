from typing import Literal

from pydantic import BaseModel, Field

type RuntimeMode = Literal["in_process", "isolated_python"]


class RuntimeBackendConfig(BaseModel):
    mode: RuntimeMode = "in_process"
    python_executable: str | None = None
    working_directory: str | None = None
    startup_timeout_seconds: int = Field(default=60, ge=1)
    gpu_memory_utilization: float | None = Field(default=None, gt=0.0, le=1.0)
    kv_cache_memory_bytes: int | None = Field(default=None, ge=1)
    max_model_len: int | None = Field(default=None, ge=1)
    max_num_seqs: int | None = Field(default=None, ge=1)
    max_num_batched_tokens: int | None = Field(default=None, ge=1)
    stage_configs_path: str | None = None


class RuntimeBackendsConfig(BaseModel):
    faster_whisper: RuntimeBackendConfig = RuntimeBackendConfig()
    kokoro: RuntimeBackendConfig = RuntimeBackendConfig()
    onnxruntime: RuntimeBackendConfig = RuntimeBackendConfig()
    vllm: RuntimeBackendConfig = RuntimeBackendConfig(
        mode="isolated_python",
        python_executable=".venv-vllm/bin/python",
        startup_timeout_seconds=300,
        gpu_memory_utilization=0.25,
        kv_cache_memory_bytes=8 * 1024 * 1024 * 1024,
        max_model_len=4096,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
    )
    vllm_omni: RuntimeBackendConfig = RuntimeBackendConfig(
        mode="isolated_python",
        python_executable=".venv-vllm/bin/python",
        startup_timeout_seconds=300,
    )

    def get_backend(self, backend_name: str | None) -> RuntimeBackendConfig | None:
        if backend_name is None:
            return None
        return getattr(self, backend_name, None)
