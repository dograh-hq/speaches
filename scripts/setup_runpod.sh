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
    apt-get install -y -qq ffmpeg espeak-ng > /dev/null
    echo "Installed ffmpeg and espeak-ng"
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

# --- Start server ---
echo ""
echo "=== Starting Speaches server on $HOST:$PORT ==="
export WHISPER__INFERENCE_DEVICE="${WHISPER__INFERENCE_DEVICE:-auto}"
exec uvicorn --factory --host "$HOST" --port "$PORT" speaches.main:create_app
