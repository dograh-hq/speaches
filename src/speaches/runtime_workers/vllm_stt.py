from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import sys
from typing import Any

from fastapi import UploadFile
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import build_async_engine_client_from_engine_args
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.speech_to_text.protocol import TranscriptionRequest
from vllm.entrypoints.openai.speech_to_text.serving import OpenAIServingTranscription
from vllm.usage.usage_lib import UsageContext

logging.basicConfig(level=logging.INFO, stream=sys.stderr)


class VllmSpeechToTextWorker:
    def __init__(
        self,
        model_path: str,
        served_model_name: str,
        gpu_memory_utilization: float | None,
        kv_cache_memory_bytes: int | None,
        max_model_len: int | None,
        max_num_seqs: int | None,
        max_num_batched_tokens: int | None,
    ) -> None:
        self.model_path = model_path
        self.served_model_name = served_model_name
        self.gpu_memory_utilization = gpu_memory_utilization
        self.kv_cache_memory_bytes = kv_cache_memory_bytes
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.engine_context = None
        self.engine_client = None
        self.transcription_service: OpenAIServingTranscription | None = None

    async def start(self) -> dict[str, Any]:
        engine_kwargs: dict[str, Any] = dict(
            model=self.model_path,
            tokenizer=self.model_path,
            tokenizer_mode="mistral",
            config_format="mistral",
            load_format="mistral",
            dtype="auto",
        )
        if self.gpu_memory_utilization is not None:
            engine_kwargs["gpu_memory_utilization"] = self.gpu_memory_utilization
        if self.kv_cache_memory_bytes is not None:
            engine_kwargs["kv_cache_memory_bytes"] = self.kv_cache_memory_bytes
        if self.max_model_len is not None:
            engine_kwargs["max_model_len"] = self.max_model_len
        if self.max_num_seqs is not None:
            engine_kwargs["max_num_seqs"] = self.max_num_seqs
        if self.max_num_batched_tokens is not None:
            engine_kwargs["max_num_batched_tokens"] = self.max_num_batched_tokens

        engine_args = AsyncEngineArgs(**engine_kwargs)
        self.engine_context = build_async_engine_client_from_engine_args(
            engine_args,
            usage_context=UsageContext.OPENAI_API_SERVER,
        )
        self.engine_client = await self.engine_context.__aenter__()
        serving_models = OpenAIServingModels(
            engine_client=self.engine_client,
            base_model_paths=[BaseModelPath(name=self.served_model_name, model_path=self.model_path)],
            lora_modules=None,
        )
        await serving_models.init_static_loras()
        self.transcription_service = OpenAIServingTranscription(
            self.engine_client,
            serving_models,
            request_logger=None,
        )
        return {"event": "ready"}

    async def transcribe(self, params: dict[str, Any]) -> dict[str, Any]:
        assert self.transcription_service is not None
        audio_bytes = base64.b64decode(params["audio_base64"])
        upload = UploadFile(filename=params.get("filename", "audio.wav"), file=io.BytesIO(audio_bytes))
        request = TranscriptionRequest(
            file=upload,
            model=self.served_model_name,
            language=params.get("language"),
            prompt=params.get("prompt", ""),
            response_format=params.get("response_format", "json"),
            temperature=params.get("temperature", 0.0),
            **{"timestamp_granularities[]": params.get("timestamp_granularities", [])},
        )
        response = await self.transcription_service.create_transcription(audio_bytes, request, raw_request=None)
        if hasattr(response, "error"):
            raise RuntimeError(response.model_dump_json())
        if hasattr(response, "__aiter__"):
            raise RuntimeError("Streaming transcription is not supported by this worker")
        return {"ok": True, "result": response.model_dump(mode="json")}

    async def shutdown(self) -> None:
        if self.engine_context is not None:
            await self.engine_context.__aexit__(None, None, None)
            self.engine_context = None
            self.engine_client = None
            self.transcription_service = None


async def _run(
    model_path: str,
    served_model_name: str,
    gpu_memory_utilization: float | None,
    kv_cache_memory_bytes: int | None,
    max_model_len: int | None,
    max_num_seqs: int | None,
    max_num_batched_tokens: int | None,
) -> int:
    worker = VllmSpeechToTextWorker(
        model_path,
        served_model_name,
        gpu_memory_utilization,
        kv_cache_memory_bytes,
        max_model_len,
        max_num_seqs,
        max_num_batched_tokens,
    )
    try:
        ready = await worker.start()
        print(json.dumps(ready), flush=True)
        for line in sys.stdin:
            request = json.loads(line)
            request_id = request.get("id")
            method = request.get("method")
            params = request.get("params", {})
            try:
                if method == "transcribe":
                    result = await worker.transcribe(params)
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
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    args = parser.parse_args()
    return asyncio.run(
        _run(
            args.model_path,
            args.served_model_name,
            args.gpu_memory_utilization,
            args.kv_cache_memory_bytes,
            args.max_model_len,
            args.max_num_seqs,
            args.max_num_batched_tokens,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
