from __future__ import annotations

import base64
from collections.abc import Generator
import logging
from pathlib import Path
import time

import huggingface_hub
import openai.types.audio
from pydantic import BaseModel

from speaches.api_types import Model
from speaches.audio import convert_audio_format
from speaches.executors.shared.base_model_manager import BaseModelManager
from speaches.executors.shared.handler_protocol import (
    NonStreamingTranscriptionResponse,
    StreamingTranscriptionEvent,
    TranscriptionRequest,
)
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
from speaches.text_utils import format_as_srt, format_as_vtt
from speaches.tracing import traced

logger = logging.getLogger(__name__)

HF_LIBRARY_NAME = "vllm"
RUNTIME_BACKEND = "vllm"
TASK_NAME_TAG = "automatic-speech-recognition"
SUPPORTED_VOXTRAL_MINI_MODEL_IDS = {"mistralai/Voxtral-Mini-3B-2507"}


class VoxtralMiniModelFiles(BaseModel):
    model_path: Path
    config_path: Path | None = None
    readme_path: Path | None = None


hf_model_filter = HfModelFilter(
    model_name="Voxtral-Mini",
    library_name=HF_LIBRARY_NAME,
    task=TASK_NAME_TAG,
)


class VoxtralMiniModelRegistry(ModelRegistry[Model, VoxtralMiniModelFiles]):
    def matches_model(self, model_id: str, model_card_data) -> bool:  # type: ignore[override]
        del model_card_data
        return model_id in SUPPORTED_VOXTRAL_MINI_MODEL_IDS

    def list_remote_models(self) -> Generator[Model]:
        for model_id in sorted(SUPPORTED_VOXTRAL_MINI_MODEL_IDS):
            model = huggingface_hub.model_info(model_id, files_metadata=False)
            if model.created_at is None:
                continue
            yield Model(
                id=model_id,
                created=int(model.created_at.timestamp()),
                owned_by=model_id.split("/")[0],
                language=[],
                task=TASK_NAME_TAG,
            )

    def list_local_models(self) -> Generator[Model]:
        for cached_repo_info in get_cached_model_repos_info():
            if cached_repo_info.repo_id not in SUPPORTED_VOXTRAL_MINI_MODEL_IDS:
                continue
            model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
            if model_card_data is None or not self.matches_model(cached_repo_info.repo_id, model_card_data):
                continue
            yield Model(
                id=cached_repo_info.repo_id,
                created=int(cached_repo_info.last_modified),
                owned_by=cached_repo_info.repo_id.split("/")[0],
                language=extract_language_list(model_card_data),
                task=TASK_NAME_TAG,
            )

    def get_model(self, model_id: str) -> Model:
        model_files = self.get_model_files(model_id)
        model_card_data = load_repo_model_card_data(model_files.readme_path) if model_files.readme_path else None
        return Model(
            id=model_id,
            created=0,
            owned_by=model_id.split("/")[0],
            language=extract_language_list(model_card_data) if model_card_data is not None else [],
            task=TASK_NAME_TAG,
        )

    def get_model_files(self, model_id: str) -> VoxtralMiniModelFiles:
        repo_path = get_model_repo_path(model_id)
        if repo_path is None:
            raise FileNotFoundError(f"Model '{model_id}' is not installed locally")
        model_files = list(list_model_files(model_id) or [])
        config_path = next((path for path in model_files if path.name == "config.json"), None)
        readme_path = next((path for path in model_files if path.name == "README.md"), None)
        model_path = config_path.parent if config_path is not None else readme_path.parent if readme_path is not None else repo_path
        return VoxtralMiniModelFiles(model_path=model_path, config_path=config_path, readme_path=readme_path)

    def download_model_files(self, model_id: str) -> None:
        huggingface_hub.snapshot_download(repo_id=model_id, repo_type="model")


voxtral_mini_model_registry = VoxtralMiniModelRegistry(
    hf_model_filter=hf_model_filter,
    hf_library_name=HF_LIBRARY_NAME,
    runtime_backend=RUNTIME_BACKEND,
)


class VoxtralMiniRuntimeSession:
    def __init__(self, runtime_config: RuntimeBackendConfig, model_id: str, model_path: Path) -> None:
        self.model_id = model_id
        module_args = ["--model-path", str(model_path), "--served-model-name", model_id]
        if runtime_config.gpu_memory_utilization is not None:
            module_args.extend(["--gpu-memory-utilization", str(runtime_config.gpu_memory_utilization)])
        if runtime_config.kv_cache_memory_bytes is not None:
            module_args.extend(["--kv-cache-memory-bytes", str(runtime_config.kv_cache_memory_bytes)])
        if runtime_config.max_model_len is not None:
            module_args.extend(["--max-model-len", str(runtime_config.max_model_len)])
        if runtime_config.max_num_seqs is not None:
            module_args.extend(["--max-num-seqs", str(runtime_config.max_num_seqs)])
        if runtime_config.max_num_batched_tokens is not None:
            module_args.extend(["--max-num-batched-tokens", str(runtime_config.max_num_batched_tokens)])
        self.client = IsolatedPythonProcessClient(
            runtime_config,
            module="speaches.runtime_workers.vllm_stt",
            module_args=module_args,
        )
        self.client.start()

    def transcribe(self, request: TranscriptionRequest) -> dict:
        wav_bytes = convert_audio_format(
            request.audio.as_bytes(),
            sample_rate=request.audio.sample_rate,
            audio_format="WAV",
        )
        filename = request.audio.name or "audio"
        response = self.client.request(
            "transcribe",
            {
                "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
                "filename": f"{filename}.wav" if not filename.endswith(".wav") else filename,
                "language": request.language,
                "prompt": request.prompt or "",
                "temperature": request.temperature,
                "response_format": request.response_format,
                "timestamp_granularities": request.timestamp_granularities,
            },
        )
        return response["result"]

    def close(self) -> None:
        self.client.close()


def _segment_dict_to_openai(segment: dict) -> openai.types.audio.TranscriptionSegment:
    return openai.types.audio.TranscriptionSegment(
        id=int(segment["id"]),
        seek=int(segment["seek"]),
        start=float(segment["start"]),
        end=float(segment["end"]),
        text=str(segment["text"]),
        tokens=[int(token) for token in segment.get("tokens", [])],
        temperature=float(segment.get("temperature", 0.0)),
        avg_logprob=float(segment.get("avg_logprob", 0.0)),
        compression_ratio=float(segment.get("compression_ratio", 0.0)),
        no_speech_prob=float(segment["no_speech_prob"]) if segment.get("no_speech_prob") is not None else None,
    )


class VoxtralMiniModelManager(BaseModelManager[VoxtralMiniRuntimeSession]):
    def __init__(self, ttl: int, runtime_config: RuntimeBackendConfig | None) -> None:
        super().__init__(ttl)
        self.runtime_config = runtime_config

    def _load_fn(self, model_id: str) -> VoxtralMiniRuntimeSession:
        if self.runtime_config is None:
            msg = "Voxtral Mini requires a runtime configuration"
            raise ValueError(msg)
        model_files = voxtral_mini_model_registry.get_model_files(model_id)
        return VoxtralMiniRuntimeSession(self.runtime_config, model_id, model_files.model_path)

    def _unload_fn(self, model: VoxtralMiniRuntimeSession) -> None:
        model.close()

    @traced()
    def handle_non_streaming_transcription_request(
        self,
        request: TranscriptionRequest,
        **_kwargs,
    ) -> NonStreamingTranscriptionResponse:
        if request.response_format not in ("text", "json", "verbose_json", "srt", "vtt"):
            raise ValueError(
                f"'{request.response_format}' response format is not supported for '{request.model}' model."
            )

        worker_response_format = "verbose_json" if request.response_format in ("verbose_json", "srt", "vtt") else request.response_format

        start = time.perf_counter()
        with self.load_model(request.model) as runtime_session:
            result = runtime_session.transcribe(request.model_copy(update={"response_format": worker_response_format}))

        logger.info(f"Transcribed {request.audio.duration} seconds of audio in {time.perf_counter() - start:.2f}s")

        if request.response_format == "text":
            return str(result["text"]), "text/plain"
        if request.response_format == "json":
            return openai.types.audio.Transcription(text=str(result["text"]))

        segments = [_segment_dict_to_openai(segment) for segment in result.get("segments", [])]
        verbose = openai.types.audio.TranscriptionVerbose(
            language=str(result.get("language") or request.language or ""),
            duration=str(result.get("duration", request.audio.duration)),
            text=str(result["text"]),
            segments=segments,
            words=None,
        )
        if request.response_format == "verbose_json":
            return verbose
        if request.response_format == "vtt":
            return "".join(format_as_vtt(segment.text, segment.start, segment.end, i) for i, segment in enumerate(segments)), "text/vtt"
        return "".join(format_as_srt(segment.text, segment.start, segment.end, i) for i, segment in enumerate(segments)), "text/plain"

    def handle_streaming_transcription_request(
        self,
        request: TranscriptionRequest,
        **_kwargs,
    ) -> Generator[StreamingTranscriptionEvent]:
        raise NotImplementedError(f"'{request.model}' model doesn't support streaming transcription.")

    def handle_transcription_request(
        self,
        request: TranscriptionRequest,
        **kwargs,
    ) -> NonStreamingTranscriptionResponse | Generator[StreamingTranscriptionEvent]:
        if request.stream:
            return self.handle_streaming_transcription_request(request, **kwargs)
        return self.handle_non_streaming_transcription_request(request, **kwargs)
