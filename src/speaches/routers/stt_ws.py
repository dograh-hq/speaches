import asyncio
import json
import logging
import time

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
    status,
)
import numpy as np
from starlette.websockets import WebSocketState

from speaches.audio import Audio
from speaches.dependencies import (
    ConfigDependency,
    ExecutorRegistryDependency,
)
from speaches.executors.shared.handler_protocol import TranscriptionRequest
from speaches.executors.silero_vad_v5 import VadOptions
from speaches.realtime.utils import verify_websocket_api_key
from speaches.routers.utils import find_executor_for_model_or_raise, get_model_card_data_or_raise

logger = logging.getLogger(__name__)

router = APIRouter(tags=["speech-to-text"])

SAMPLE_RATE = 16000


def _transcribe(
    audio_data: np.typing.NDArray[np.float32],
    model: str,
    language: str | None,
    executor_registry: "ExecutorRegistryDependency",
) -> tuple[str, float]:
    audio = Audio(data=audio_data, sample_rate=SAMPLE_RATE)
    model_card_data = get_model_card_data_or_raise(model)
    executor = find_executor_for_model_or_raise(model, model_card_data, executor_registry.transcription)
    request = TranscriptionRequest(
        audio=audio,
        model=model,
        language=language,
        response_format="text",
        speech_segments=[],
        vad_options=VadOptions(),
        timestamp_granularities=["segment"],
    )
    start = time.perf_counter()
    result = executor.model_manager.handle_transcription_request(request)
    elapsed = time.perf_counter() - start
    text = result[0] if isinstance(result, tuple) else str(result)
    return text, elapsed


@router.websocket("/v1/stt/ws")
async def stt_stream(
    ws: WebSocket,
    config: ConfigDependency,
    executor_registry: ExecutorRegistryDependency,
) -> None:
    try:
        await verify_websocket_api_key(ws, config)
    except WebSocketException:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication failed")
        return

    await ws.accept()

    model: str | None = None
    language: str | None = None
    sample_rate: int = SAMPLE_RATE
    audio_buffer: np.typing.NDArray[np.float32] = np.array([], dtype=np.float32)

    try:
        while True:
            message = await ws.receive()

            if message["type"] == "websocket.disconnect":
                break

            # Binary frame: raw audio
            if message.get("bytes"):
                raw_bytes = message["bytes"]
                audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
                audio_float = audio_int16.astype(np.float32) / 32768.0
                if sample_rate != SAMPLE_RATE:
                    from speaches.audio import resample_audio_data

                    audio_float = resample_audio_data(audio_float, sample_rate, SAMPLE_RATE)
                audio_buffer = np.append(audio_buffer, audio_float)
                continue

            # Text frame: JSON control message
            if message.get("text"):
                try:
                    msg = json.loads(message["text"])
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                msg_type = msg.get("type")

                if msg_type == "config":
                    model = msg.get("model", model)
                    language = msg.get("language", language)
                    sample_rate = msg.get("sample_rate", sample_rate)
                    logger.info(f"STT WS configured: model={model}, language={language}, sample_rate={sample_rate}")
                    await ws.send_json({"type": "ready"})

                elif msg_type == "finalize":
                    if model is None:
                        await ws.send_json(
                            {"type": "error", "message": "No model configured. Send a config message first."}
                        )
                        continue
                    if len(audio_buffer) < SAMPLE_RATE * 0.1:
                        # Less than 100ms of audio, send empty final
                        await ws.send_json(
                            {
                                "type": "transcription",
                                "text": "",
                                "is_final": True,
                                "from_finalize": True,
                            }
                        )
                        audio_buffer = np.array([], dtype=np.float32)
                        continue

                    text, elapsed = await asyncio.to_thread(
                        _transcribe, audio_buffer, model, language, executor_registry
                    )
                    logger.info(
                        f"Transcription took {elapsed:.2f}s for {len(audio_buffer) / SAMPLE_RATE:.2f}s of audio"
                    )
                    await ws.send_json(
                        {
                            "type": "transcription",
                            "text": text.strip(),
                            "is_final": True,
                            "from_finalize": True,
                        }
                    )
                    audio_buffer = np.array([], dtype=np.float32)

                elif msg_type == "keepalive":
                    pass

                elif msg_type == "end_of_stream":
                    logger.info("Received end_of_stream, closing connection")
                    break

                else:
                    await ws.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info("STT WebSocket client disconnected")
    except Exception:
        logger.exception("Error in STT WebSocket handler")
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.close(code=status.WS_1011_INTERNAL_ERROR)
