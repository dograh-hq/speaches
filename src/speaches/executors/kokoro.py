from collections.abc import Generator
import logging
from typing import Literal

import huggingface_hub
from pydantic import BaseModel, computed_field

from speaches.api_types import (
    OPENAI_SUPPORTED_SPEECH_VOICE_NAMES,
    Model,
)
from speaches.audio import Audio
from speaches.executors.shared.base_model_manager import BaseModelManager
from speaches.executors.shared.handler_protocol import SpeechRequest, SpeechResponse
from speaches.hf_utils import (
    HfModelFilter,
    extract_language_list,
    get_cached_model_repos_info,
    get_model_card_data_from_cached_repo_info,
)
from speaches.model_registry import (
    ModelRegistry,
)
from speaches.tracing import traced_generator

SAMPLE_RATE = 24000  # the default sample rate for Kokoro
TASK_NAME_TAG = "text-to-speech"

# HuggingFace model ID for the PyTorch Kokoro model
PYTORCH_MODEL_ID = "hexgrad/Kokoro-82M"


class KokoroModelVoice(BaseModel):
    name: str
    language: str
    gender: Literal["male", "female"]

    @computed_field
    @property
    def id(self) -> str:
        return self.name


VOICES = [
    # American English
    KokoroModelVoice(name="af_heart", language="en-us", gender="female"),
    KokoroModelVoice(name="af_alloy", language="en-us", gender="female"),
    KokoroModelVoice(name="af_aoede", language="en-us", gender="female"),
    KokoroModelVoice(name="af_bella", language="en-us", gender="female"),
    KokoroModelVoice(name="af_jessica", language="en-us", gender="female"),
    KokoroModelVoice(name="af_kore", language="en-us", gender="female"),
    KokoroModelVoice(name="af_nicole", language="en-us", gender="female"),
    KokoroModelVoice(name="af_nova", language="en-us", gender="female"),
    KokoroModelVoice(name="af_river", language="en-us", gender="female"),
    KokoroModelVoice(name="af_sarah", language="en-us", gender="female"),
    KokoroModelVoice(name="af_sky", language="en-us", gender="female"),
    KokoroModelVoice(name="am_adam", language="en-us", gender="male"),
    KokoroModelVoice(name="am_echo", language="en-us", gender="male"),
    KokoroModelVoice(name="am_eric", language="en-us", gender="male"),
    KokoroModelVoice(name="am_fenrir", language="en-us", gender="male"),
    KokoroModelVoice(name="am_liam", language="en-us", gender="male"),
    KokoroModelVoice(name="am_michael", language="en-us", gender="male"),
    KokoroModelVoice(name="am_onyx", language="en-us", gender="male"),
    KokoroModelVoice(name="am_puck", language="en-us", gender="male"),
    KokoroModelVoice(name="am_santa", language="en-us", gender="male"),
    # British English
    KokoroModelVoice(name="bf_alice", language="en-gb", gender="female"),
    KokoroModelVoice(name="bf_emma", language="en-gb", gender="female"),
    KokoroModelVoice(name="bf_isabella", language="en-gb", gender="female"),
    KokoroModelVoice(name="bf_lily", language="en-gb", gender="female"),
    KokoroModelVoice(name="bm_daniel", language="en-gb", gender="male"),
    KokoroModelVoice(name="bm_fable", language="en-gb", gender="male"),
    KokoroModelVoice(name="bm_george", language="en-gb", gender="male"),
    KokoroModelVoice(name="bm_lewis", language="en-gb", gender="male"),
    # Japanese
    KokoroModelVoice(name="jf_alpha", language="ja", gender="female"),
    KokoroModelVoice(name="jf_gongitsune", language="ja", gender="female"),
    KokoroModelVoice(name="jf_nezumi", language="ja", gender="female"),
    KokoroModelVoice(name="jf_tebukuro", language="ja", gender="female"),
    KokoroModelVoice(name="jm_kumo", language="ja", gender="male"),
    # Mandarin Chinese
    KokoroModelVoice(name="zf_xiaobei", language="zh", gender="female"),
    KokoroModelVoice(name="zf_xiaoni", language="zh", gender="female"),
    KokoroModelVoice(name="zf_xiaoxiao", language="zh", gender="female"),
    KokoroModelVoice(name="zf_xiaoyi", language="zh", gender="female"),
    KokoroModelVoice(name="zm_yunjian", language="zh", gender="male"),
    KokoroModelVoice(name="zm_yunxi", language="zh", gender="male"),
    KokoroModelVoice(name="zm_yunxia", language="zh", gender="male"),
    KokoroModelVoice(name="zm_yunyang", language="zh", gender="male"),
    # Spanish
    KokoroModelVoice(name="ef_dora", language="es", gender="female"),
    KokoroModelVoice(name="em_alex", language="es", gender="male"),
    KokoroModelVoice(name="em_santa", language="es", gender="male"),
    # French
    KokoroModelVoice(name="ff_siwis", language="fr-fr", gender="female"),
    # Hindi
    KokoroModelVoice(name="hf_alpha", language="hi", gender="female"),
    KokoroModelVoice(name="hf_beta", language="hi", gender="female"),
    KokoroModelVoice(name="hm_omega", language="hi", gender="male"),
    KokoroModelVoice(name="hm_psi", language="hi", gender="male"),
    # Italian
    KokoroModelVoice(name="if_sara", language="it", gender="female"),
    KokoroModelVoice(name="im_nicola", language="it", gender="male"),
    # Brazilian Portuguese
    KokoroModelVoice(name="pf_dora", language="pt-br", gender="female"),
    KokoroModelVoice(name="pm_alex", language="pt-br", gender="male"),
    KokoroModelVoice(name="pm_santa", language="pt-br", gender="male"),
]

# Map voice language codes to Kokoro KPipeline lang_code
LANG_CODE_MAP = {
    "en-us": "a",
    "en-gb": "b",
    "ja": "j",
    "zh": "z",
    "es": "e",
    "fr-fr": "f",
    "hi": "h",
    "it": "i",
    "pt-br": "p",
}


class KokoroModel(Model):
    sample_rate: int
    voices: list[KokoroModelVoice]


# Use a broader filter that matches the PyTorch model on HuggingFace
hf_model_filter = HfModelFilter(
    library_name="kokoro",
    task=TASK_NAME_TAG,
    tags={"kokoro"},
)


logger = logging.getLogger(__name__)


class KokoroModelRegistry(ModelRegistry):
    def list_remote_models(self) -> Generator[KokoroModel]:
        # The PyTorch Kokoro model doesn't match the ONNX HF filter,
        # so we hardcode the known model.
        yield KokoroModel(
            id=PYTORCH_MODEL_ID,
            created=0,
            owned_by="hexgrad",
            language=["en", "ja", "zh", "es", "fr", "hi", "it", "pt"],
            task=TASK_NAME_TAG,
            sample_rate=SAMPLE_RATE,
            voices=VOICES,
        )

    def list_local_models(self) -> Generator[KokoroModel]:
        cached_model_repos_info = get_cached_model_repos_info()
        for cached_repo_info in cached_model_repos_info:
            if cached_repo_info.repo_id == PYTORCH_MODEL_ID:
                model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
                yield KokoroModel(
                    id=cached_repo_info.repo_id,
                    created=int(cached_repo_info.last_modified),
                    owned_by=cached_repo_info.repo_id.split("/")[0],
                    language=extract_language_list(model_card_data) if model_card_data else None,
                    task=TASK_NAME_TAG,
                    sample_rate=SAMPLE_RATE,
                    voices=VOICES,
                )

    def get_model_files(self, model_id: str):
        # KPipeline handles model file resolution internally
        return None

    def download_model_files(self, model_id: str) -> None:
        _model_repo_path_str = huggingface_hub.snapshot_download(
            repo_id=model_id, repo_type="model"
        )


kokoro_model_registry = KokoroModelRegistry(hf_model_filter=hf_model_filter)


class KokoroModelManager(BaseModelManager):
    def __init__(self, ttl: int) -> None:
        super().__init__(ttl)

    def _load_fn(self, model_id: str):
        from kokoro import KPipeline

        # Default to American English; voice-specific lang is set per request
        return KPipeline(lang_code="a")

    @traced_generator()
    def handle_speech_request(
        self,
        request: SpeechRequest,
        **_kwargs,
    ) -> SpeechResponse:
        if request.speed < 0.25 or request.speed > 4.0:
            msg = f"Speed must be between 0.25 and 4.0, got {request.speed}"
            raise ValueError(msg)
        if request.voice not in [v.name for v in VOICES]:
            if request.voice in OPENAI_SUPPORTED_SPEECH_VOICE_NAMES:
                logger.warning(
                    f"Voice '{request.voice}' is not supported by the model '{request.model}'. It will be replaced with '{VOICES[0].name}'."
                )
                request.voice = VOICES[0].name
            else:
                msg = f"Voice '{request.voice}' is not supported. Supported voices: {[v.name for v in VOICES]}"
                raise ValueError(msg)

        voice_language = next(v.language for v in VOICES if v.name == request.voice)
        lang_code = LANG_CODE_MAP.get(voice_language, "a")

        import time

        with self.load_model(request.model) as pipeline:
            start = time.perf_counter()
            # KPipeline yields (graphemes, phonemes, audio_numpy) tuples
            for _gs, _ps, audio_data in pipeline(
                request.text,
                voice=request.voice,
                speed=request.speed,
                lang_code=lang_code,
                split_pattern=r"\n+",
            ):
                if audio_data is not None and len(audio_data) > 0:
                    yield Audio(audio_data, sample_rate=SAMPLE_RATE)

        logger.info(f"Generated audio for {len(request.text)} characters in {time.perf_counter() - start:.2f}s")
