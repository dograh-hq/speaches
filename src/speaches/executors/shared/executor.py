from huggingface_hub import ModelCardData
from pydantic import BaseModel

from speaches.api_types import ModelTask
from speaches.model_registry import ModelRegistry
from speaches.runtime import RuntimeBackendConfig, RuntimeMode


class Executor[ManagerT, RegistryT: ModelRegistry](BaseModel):
    name: str
    model_manager: ManagerT
    model_registry: RegistryT
    task: ModelTask
    runtime_config: RuntimeBackendConfig | None = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def hf_library_name(self) -> str | None:
        return self.model_registry.hf_library_name

    @property
    def runtime_backend(self) -> str | None:
        return self.model_registry.runtime_backend

    @property
    def runtime_mode(self) -> RuntimeMode | None:
        if self.runtime_config is None:
            return None
        return self.runtime_config.mode

    def can_handle_model(self, model_id: str, model_card_data: ModelCardData) -> bool:
        return self.model_registry.matches_model(model_id, model_card_data)
