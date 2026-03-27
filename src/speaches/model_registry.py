from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

from speaches.api_types import Model
from speaches.hf_utils import (
    HfModelFilter,
)

if TYPE_CHECKING:
    from huggingface_hub import ModelCardData


class ModelRegistry[ModelT: Model, ModelFilesT]:
    def __init__(
        self,
        hf_model_filter: HfModelFilter,
        *,
        hf_library_name: str | None = None,
        runtime_backend: str | None = None,
    ) -> None:
        self.hf_model_filter = hf_model_filter
        self.hf_library_name = hf_library_name if hf_library_name is not None else hf_model_filter.library_name
        self.runtime_backend = runtime_backend if runtime_backend is not None else self.hf_library_name

    def matches_model(self, model_id: str, model_card_data: ModelCardData) -> bool:
        return self.hf_model_filter.passes_filter(model_id, model_card_data)

    def list_remote_models(self) -> Generator[ModelT, None]: ...
    def list_local_models(self) -> Generator[ModelT, None]: ...
    def get_model(self, model_id: str) -> ModelT: ...
    def get_model_files(self, model_id: str) -> ModelFilesT: ...
    def download_model_files(self, model_id: str) -> None: ...
    def download_model_files_if_not_exist(self, model_id: str) -> bool:
        try:
            self.get_model_files(model_id)
        except Exception:  # noqa: BLE001
            self.download_model_files(model_id)
            return True
        return False
