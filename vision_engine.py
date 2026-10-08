"""
Vision module: captures a frame from an IP Webcam / DroidCam stream and asks
a local Vision-Language Model (via Ollama) a question about it.
"""
import os
import base64
import logging

import cv2
import requests

from config import CAMERA_STREAM_URL, VLM_MODEL

logger = logging.getLogger(__name__)

# Trigger words that activate vision mode (kept in Italian: they are matched
# against the user's spoken Italian input, same pattern as rag_engine.py).
VISION_TRIGGER_WORDS = ["guarda", "vedi", "dov'è", "cosa c'è", "mostra"]


def is_vision_request(text: str) -> bool:
    """True if the spoken text contains one of the vision trigger words."""
    text_lower = text.lower()
    return any(trigger in text_lower for trigger in VISION_TRIGGER_WORDS)


def capture_frame() -> str | None:
    """Grabs a single frame from the camera stream and saves it to a temp file.
    Returns the file path, or None if the stream could not be read."""
    logger.info("Capturing frame from camera stream...")
    cap = cv2.VideoCapture(CAMERA_STREAM_URL)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        logger.error("Could not read a frame from the camera stream.")
        return None

    temp_path = "temp_frame.jpg"
    cv2.imwrite(temp_path, frame)
    return temp_path


def query_vlm(prompt: str, image_path: str) -> str:
    """Sends the image and the user's actual question to the local VLM via Ollama."""
    logger.info("Sending image and prompt to local VLM...")
    with open(image_path, "rb") as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode("utf-8")

    payload = {
        "model": VLM_MODEL,
        # IMPORTANT: use the user's actual question, not a fixed generic prompt.
        # Instructing the model to answer in Italian keeps it consistent with
        # the rest of the assistant (spoken responses, conversation history).
        "prompt": f"Rispondi in italiano. Domanda dell'utente: {prompt}",
        "images": [img_b64],
        "stream": False,
        "keep_alive": 0,  # unload the VLM right after use to free VRAM for the LLM
        "options": {
            "temperature": 0.4,
            "repeat_penalty": 1.3,  # mitigates repetition loops some VLM builds are prone to
        },
    }

    try:
        response = requests.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=60)
        response.raise_for_status()

        if os.path.exists(image_path):
            os.remove(image_path)

        return response.json().get("response", "Errore nella risposta del modello visivo.")
    except requests.RequestException as e:
        logger.error(f"Ollama connection error: {e}")
        return f"Errore di connessione a Ollama: {e}"
