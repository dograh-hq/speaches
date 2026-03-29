from __future__ import annotations

import base64
from collections.abc import Generator
import io
import json
import logging
from pathlib import Path
import time

import huggingface_hub
import soundfile as sf
from pydantic import BaseModel, computed_field

from speaches.api_types import Model
from speaches.audio import Audio
from speaches.executors.shared.base_model_manager import BaseModelManager
from speaches.executors.shared.handler_protocol import SpeechRequest, SpeechResponse
from speaches.hf_utils import (
    HfModelFilter,
    extract_language_list,
    get_cached_model_repos_info,
    get_model_card_data_from_cached_repo_info,
    get_model_repo_path,
    list_model_files,
    load_repo_model_card_data,
)
from speaches.model_registry import ModelRegistry
from speaches.runtime import RuntimeBackendConfig
from speaches.runtime_backends import IsolatedPythonProcessClient
from speaches.tracing import traced_generator

logger = logging.getLogger(__name__)

HF_LIBRARY_NAME = "vllm"
RUNTIME_BACKEND = "vllm_omni"
TASK_NAME_TAG = "text-to-speech"
DEFAULT_SAMPLE_RATE = 24000
SUPPORTED_VOXTRAL_MODEL_IDS = {"mistralai/Voxtral-4B-TTS-2603"}
DEFAULT_STAGE_CONFIG_PATH = "configuration/vllm_omni/voxtral_tts_low_memory.yaml"


class VoxtralModelFiles(BaseModel):
    model_path: Path
    config_path: Path | None = None
    readme_path: Path | None = None


class VoxtralModelVoice(BaseModel):
    name: str

    @computed_field
    @property
    def id(self) -> str:
        return self.name


class VoxtralModel(Model):
    sample_rate: int = DEFAULT_SAMPLE_RATE
    voices: list[VoxtralModelVoice]


hf_model_filter = HfModelFilter(
    model_name="Voxtral",
    library_name=HF_LIBRARY_NAME,
    task=TASK_NAME_TAG,
)


def _load_voices_and_sample_rate(config_path: Path | None) -> tuple[list[VoxtralModelVoice], int]:
    if config_path is None or not config_path.exists():
        return [], DEFAULT_SAMPLE_RATE

    config = json.loads(config_path.read_text())
    audio_config = config.get("audio_config", {})
    speaker_map = audio_config.get("spk_id") or audio_config.get("speaker_id") or {}
    voices = [VoxtralModelVoice(name=speaker.lower()) for speaker in sorted(speaker_map.keys())]
    sample_rate = int(audio_config.get("sampling_rate", DEFAULT_SAMPLE_RATE))
    return voices, sample_rate


class VoxtralModelRegistry(ModelRegistry[VoxtralModel, VoxtralModelFiles]):
    def matches_model(self, model_id: str, model_card_data) -> bool:  # type: ignore[override]
        return model_id in SUPPORTED_VOXTRAL_MODEL_IDS and super().matches_model(model_id, model_card_data)

    def list_remote_models(self) -> Generator[VoxtralModel]:
        for model in huggingface_hub.list_models(**self.hf_model_filter.list_model_kwargs(), cardData=True):
            if model.id not in SUPPORTED_VOXTRAL_MODEL_IDS:
                continue
            if model.created_at is None:
                continue
            yield VoxtralModel(
                id=model.id,
                created=int(model.created_at.timestamp()),
                owned_by=model.id.split("/")[0],
                language=["en"],
                task=TASK_NAME_TAG,
                voices=[],
                sample_rate=DEFAULT_SAMPLE_RATE,
            )

    def list_local_models(self) -> Generator[VoxtralModel]:
        for cached_repo_info in get_cached_model_repos_info():
            if cached_repo_info.repo_id not in SUPPORTED_VOXTRAL_MODEL_IDS:
                continue
            model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
            if model_card_data is None or not self.matches_model(cached_repo_info.repo_id, model_card_data):
                continue
            model_files = self.get_model_files(cached_repo_info.repo_id)
            voices, sample_rate = _load_voices_and_sample_rate(model_files.config_path)
            yield VoxtralModel(
                id=cached_repo_info.repo_id,
                created=int(cached_repo_info.last_modified),
                owned_by=cached_repo_info.repo_id.split("/")[0],
                language=["en"],
                task=TASK_NAME_TAG,
                voices=voices,
                sample_rate=sample_rate,
            )

    def get_model(self, model_id: str) -> VoxtralModel:
        model_files = self.get_model_files(model_id)
        voices, sample_rate = _load_voices_and_sample_rate(model_files.config_path)
        repo_readme = model_files.readme_path
        model_card_data = load_repo_model_card_data(repo_readme) if repo_readme is not None and repo_readme.exists() else None
        return VoxtralModel(
            id=model_id,
            created=0,
            owned_by=model_id.split("/")[0],
            language=extract_language_list(model_card_data) if model_card_data is not None else ["en"],
            task=TASK_NAME_TAG,
            voices=voices,
            sample_rate=sample_rate,
        )

    def get_model_files(self, model_id: str) -> VoxtralModelFiles:
        repo_path = get_model_repo_path(model_id)
        if repo_path is None:
            raise FileNotFoundError(f"Model '{model_id}' is not installed locally")
        model_files = list(list_model_files(model_id) or [])
        config_path = next((path for path in model_files if path.name == "config.json"), None)
        readme_path = next((path for path in model_files if path.name == "README.md"), None)
        model_path = config_path.parent if config_path is not None else readme_path.parent if readme_path is not None else repo_path
        return VoxtralModelFiles(model_path=model_path, config_path=config_path, readme_path=readme_path)

    def download_model_files(self, model_id: str) -> None:
        huggingface_hub.snapshot_download(repo_id=model_id, repo_type="model")


voxtral_model_registry = VoxtralModelRegistry(
    hf_model_filter=hf_model_filter,
    hf_library_name=HF_LIBRARY_NAME,
    runtime_backend=RUNTIME_BACKEND,
)


class VoxtralRuntimeSession:
    def __init__(self, runtime_config: RuntimeBackendConfig, model_id: str, model_path: Path) -> None:
        self.model_id = model_id
        module_args = ["--model", str(model_path)]
        stage_configs_path = runtime_config.stage_configs_path or DEFAULT_STAGE_CONFIG_PATH
        module_args.extend(["--stage-configs-path", stage_configs_path])
        self.client = IsolatedPythonProcessClient(
            runtime_config,
            module="speaches.runtime_workers.vllm_omni_tts",
            module_args=module_args,
        )
        ready_message = self.client.start()
        logger.debug(f"ready_message: {ready_message}")
        self.sample_rate = int(ready_message.get("sample_rate", DEFAULT_SAMPLE_RATE))
        self.voices = [str(voice) for voice in ready_message.get("voices", [])]

    def synthesize(self, request: SpeechRequest) -> Audio:
        response = self.client.request(
            "synthesize",
            {
                "text": request.text,
                "voice": request.voice,
                "speed": request.speed,
            },
        )
        audio_bytes = base64.b64decode(response["audio_base64"])
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        return Audio(audio_data, sample_rate=int(sample_rate))

    def close(self) -> None:
        self.client.close()


class VoxtralModelManager(BaseModelManager[VoxtralRuntimeSession]):
    def __init__(self, ttl: int, runtime_config: RuntimeBackendConfig | None) -> None:
        super().__init__(ttl)
        self.runtime_config = runtime_config

    def _load_fn(self, model_id: str) -> VoxtralRuntimeSession:
        if self.runtime_config is None:
            msg = "Voxtral requires a runtime configuration"
            raise ValueError(msg)
        model_files = voxtral_model_registry.get_model_files(model_id)
        return VoxtralRuntimeSession(self.runtime_config, model_id, model_files.model_path)

    def _unload_fn(self, model: VoxtralRuntimeSession) -> None:
        model.close()

    @traced_generator()
    def handle_speech_request(self, request: SpeechRequest, **_kwargs) -> SpeechResponse:
        start = time.perf_counter()
        with self.load_model(request.model) as runtime_session:
            yield runtime_session.synthesize(request)
        logger.info(f"Generated audio for {len(request.text)} characters in {time.perf_counter() - start:.2f}s")
