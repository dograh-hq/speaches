from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
from pathlib import Path
import sys
from typing import Any

from fastapi.responses import Response
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm_omni.entrypoints import AsyncOmni
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

logging.basicConfig(level=logging.INFO, stream=sys.stderr)

LOW_MEMORY_STAGE_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configuration/vllm_omni/voxtral_tts_low_memory.yaml"


class VoxtralTtsWorker:
    def __init__(self, model: str, stage_configs_path: str | None) -> None:
        self.model = model
        self.stage_configs_path = stage_configs_path
        self.engine: AsyncOmni | None = None
        self.speech_service: OmniOpenAIServingSpeech | None = None

    async def start(self) -> dict[str, Any]:
        resolved_stage_configs_path = self.stage_configs_path or str(LOW_MEMORY_STAGE_CONFIG_PATH)
        self.engine = AsyncOmni(
            model=self.model,
            dtype="bfloat16",
            stage_configs_path=resolved_stage_configs_path,
        )
        serving_models = OpenAIServingModels(
            engine_client=self.engine,
            base_model_paths=[BaseModelPath(name=self.model, model_path=self.model)],
            lora_modules=None,
        )
        await serving_models.init_static_loras()
        self.speech_service = OmniOpenAIServingSpeech(self.engine, serving_models, request_logger=None)
        sample_rate = _extract_default_sample_rate(self.engine)
        voices = sorted(getattr(self.speech_service, "supported_speakers", set()))
        return {
            "event": "ready",
            "sample_rate": sample_rate,
            "voices": voices,
        }

    async def synthesize(self, params: dict[str, Any]) -> dict[str, Any]:
        assert self.speech_service is not None
        request = OpenAICreateSpeechRequest(
            input=params["text"],
            model=self.model,
            voice=params.get("voice"),
            response_format="wav",
            speed=params.get("speed", 1.0),
            stream=False,
        )
        response = await self.speech_service.create_speech(request)
        if not isinstance(response, Response):
            msg = f"Unexpected response type from vllm_omni: {type(response).__name__}"
            raise RuntimeError(msg)
        if response.status_code >= 400:
            detail = response.body.decode("utf-8", errors="replace")
            raise RuntimeError(detail)
        return {
            "ok": True,
            "audio_base64": base64.b64encode(response.body).decode("ascii"),
            "media_type": response.media_type,
        }

    async def shutdown(self) -> None:
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None
            self.speech_service = None


def _extract_default_sample_rate(engine: AsyncOmni) -> int:
    hf_config = engine.model_config.hf_config
    audio_config = getattr(hf_config, "audio_config", None)
    if isinstance(audio_config, dict):
        sample_rate = audio_config.get("sampling_rate")
    else:
        sample_rate = getattr(audio_config, "sampling_rate", None)
    if sample_rate is None:
        return 24000
    return int(sample_rate)


async def _run(model: str, stage_configs_path: str | None) -> int:
    worker = VoxtralTtsWorker(model, stage_configs_path)
    try:
        ready = await worker.start()
        print(json.dumps(ready), flush=True)
        for line in sys.stdin:
            request = json.loads(line)
            request_id = request.get("id")
            method = request.get("method")
            params = request.get("params", {})
            try:
                if method == "synthesize":
                    result = await worker.synthesize(params)
                elif method == "shutdown":
                    await worker.shutdown()
                    print(json.dumps({"id": request_id, "ok": True, "result": {"shutdown": True}}), flush=True)
                    return 0
                else:
                    raise ValueError(f"Unsupported worker method '{method}'")
                print(json.dumps({"id": request_id, **result}), flush=True)
            except Exception as error:  # noqa: BLE001
                print(json.dumps({"id": request_id, "ok": False, "error": str(error)}), flush=True)
    except Exception as error:  # noqa: BLE001
        print(json.dumps({"event": "error", "error": str(error)}), flush=True)
        return 1
    finally:
        await worker.shutdown()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--stage-configs-path")
    args = parser.parse_args()
    return asyncio.run(_run(args.model, args.stage_configs_path))


if __name__ == "__main__":
    raise SystemExit(main())
