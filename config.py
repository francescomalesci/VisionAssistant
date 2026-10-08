"""
Central configuration for the voice assistant.
Override any value via environment variables (see README) instead of editing
this file directly, especially the personal paths and local network URLs.
"""
import os

# --- faster-whisper CUDA DLLs (Windows only) ---
# Required because faster-whisper's bundled cuBLAS/cuDNN wheels are unreliable
# on Windows; these DLLs come from the Faster-Whisper-XXL standalone release.
# See README for where to download them.
FASTER_WHISPER_DLL_DIR = os.getenv(
    "FASTER_WHISPER_DLL_DIR",
    r"C:\path\to\Faster-Whisper-XXL\_xxl_data\torch\lib"
)
# --- Models ---
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen2.5vl:7b")
WAKE_WORD_MODEL = os.getenv("WAKE_WORD_MODEL", "alexa")
WAKE_WORD_THRESHOLD = float(os.getenv("WAKE_WORD_THRESHOLD", "0.75"))

# --- Piper TTS (Italian voice) ---
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "it_IT-serena-medium.onnx")
PIPER_CONFIG_PATH = os.getenv("PIPER_CONFIG_PATH", "it_IT-serena-medium.onnx.json")

# --- Vision: IP Webcam / DroidCam stream URL ---
# Update with your phone's current local IP (changes if it reconnects to Wi-Fi).
CAMERA_STREAM_URL = os.getenv("CAMERA_STREAM_URL", "http://192.168.1.102:8081/video")

# --- Web UI bridge (server.py) ---
UI_SERVER_URL = os.getenv("UI_SERVER_URL", "http://127.0.0.1:8000")
