#!/usr/bin/env bash
set -euo pipefail

# Speaches RunPod Setup Script
# Sets up venv, installs deps, downloads models, and starts the server.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --- Configuration ---
STT_MODEL="${STT_MODEL:-Systran/faster-distil-whisper-small.en}"
TTS_MODEL="${TTS_MODEL:-hexgrad/Kokoro-82M}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "=== Speaches RunPod Setup ==="
echo "Project dir: $PROJECT_DIR"
echo "STT model:   $STT_MODEL"
echo "TTS model:   $TTS_MODEL"

# --- System dependencies ---
echo ""
echo "--- Installing system dependencies ---"
if command -v apt-get &>/dev/null; then
    apt-get update -qq
    apt-get install -y -qq ffmpeg espeak-ng libcudnn9-cuda-12 > /dev/null 2>&1 || \
        apt-get install -y -qq ffmpeg espeak-ng > /dev/null
    echo "Installed ffmpeg, espeak-ng, and cuDNN (if available)"
else
    echo "WARNING: apt-get not available. Ensure ffmpeg and espeak-ng are installed."
fi

# --- Python environment ---
echo ""
echo "--- Setting up Python environment ---"
cd "$PROJECT_DIR"

if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

echo "Installing dependencies..."
source .venv/bin/activate
uv sync

# Install PyTorch with CUDA support (uv sync installs CPU-only by default)
if command -v nvidia-smi &>/dev/null; then
    echo "NVIDIA GPU detected, installing PyTorch with CUDA..."
    uv pip install torch --index-url https://download.pytorch.org/whl/cu126
else
    echo "No NVIDIA GPU detected, using CPU PyTorch"
fi

# --- spaCy model (needed by Kokoro's phonemizer) ---
# spacy.cli.download uses pip internally which doesn't exist in uv venvs.
# Install the model wheel directly via uv.
echo "Installing spaCy English model..."
uv pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
echo "spaCy model ready"

# --- HuggingFace cache ---
mkdir -p "${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"

# --- Download models ---
echo ""
echo "--- Downloading models ---"
echo "Downloading STT model: $STT_MODEL"
python -c "
import huggingface_hub
huggingface_hub.snapshot_download(
    repo_id='$STT_MODEL',
    repo_type='model',
    allow_patterns=['config.json', 'preprocessor_config.json', 'model.bin', 'tokenizer.json', 'vocabulary.*', 'README.md'],
)
print('STT model downloaded successfully')
"

echo "Downloading TTS model: $TTS_MODEL"
python -c "
import huggingface_hub
huggingface_hub.snapshot_download(repo_id='$TTS_MODEL', repo_type='model')
print('TTS model downloaded successfully')
"

# --- Start server and warm up models ---
echo ""
echo "=== Starting Speaches server on $HOST:$PORT ==="
# CTranslate2 (faster-whisper) needs cuDNN for GPU. Use CPU if cuDNN is missing.
if ldconfig -p 2>/dev/null | grep -q libcudnn; then
    export WHISPER__INFERENCE_DEVICE="${WHISPER__INFERENCE_DEVICE:-auto}"
    echo "cuDNN found, STT will use GPU"
else
    export WHISPER__INFERENCE_DEVICE="${WHISPER__INFERENCE_DEVICE:-cpu}"
    echo "cuDNN not found, STT will use CPU (install libcudnn9-cuda-12 for GPU)"
fi
export PRELOAD_MODELS="[\"$STT_MODEL\",\"$TTS_MODEL\"]"
# Keep models loaded in GPU memory permanently
export STT_MODEL_TTL=-1
export TTS_MODEL_TTL=-1

# Start server in background, warm up, then foreground
uvicorn --factory --host "$HOST" --port "$PORT" speaches.main:create_app &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to start..."
for i in $(seq 1 30); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server is ready"
        break
    fi
    sleep 1
done

# Warm up TTS (loads Kokoro into GPU)
echo "Warming up TTS model..."
curl -s -X POST "http://localhost:$PORT/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d "{\"input\": \"warm up\", \"model\": \"$TTS_MODEL\", \"voice\": \"af_heart\", \"response_format\": \"pcm\"}" \
  -o /dev/null 2>&1 && echo "TTS model loaded" || echo "TTS warmup failed"

# Warm up STT (loads Whisper into GPU)
echo "Warming up STT model..."
python -c "
import wave, struct, io
# Generate a tiny WAV file
buf = io.BytesIO()
with wave.open(buf, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(struct.pack('<' + 'h' * 16000, *([0] * 16000)))
buf.seek(0)
open('/tmp/_warmup.wav', 'wb').write(buf.read())
"
curl -s -X POST "http://localhost:$PORT/v1/audio/transcriptions" \
  -F "file=@/tmp/_warmup.wav" \
  -F "model=$STT_MODEL" \
  -o /dev/null 2>&1 && echo "STT model loaded" || echo "STT warmup failed"
rm -f /tmp/_warmup.wav

echo ""
echo "=== Models loaded into GPU. Server running (PID: $SERVER_PID) ==="
# Bring server back to foreground
wait $SERVER_PID
