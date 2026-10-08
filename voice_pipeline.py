"""
Main voice assistant loop: wake word detection -> VAD-based recording ->
speech-to-text (faster-whisper) -> routing (plain chat / file search /
vision) -> local LLM -> text-to-speech (Piper).

Spoken responses, trigger words and prompts to the LLM are in Italian on
purpose: this assistant is designed to be used in Italian.
"""
import os

from config import (
    FASTER_WHISPER_DLL_DIR, WHISPER_MODEL_SIZE, LLM_MODEL,
    WAKE_WORD_MODEL, WAKE_WORD_THRESHOLD,
    PIPER_MODEL_PATH, PIPER_CONFIG_PATH, UI_SERVER_URL,
)

# faster-whisper on Windows needs these CUDA DLLs explicitly added to the
# search path before faster_whisper/ctranslate2 is imported.
os.add_dll_directory(FASTER_WHISPER_DLL_DIR)
os.environ["PATH"] = FASTER_WHISPER_DLL_DIR + os.pathsep + os.environ["PATH"]

import io
import wave
import logging

import pyaudio
import numpy as np
import torch
import openwakeword
import ollama
import requests
from piper import PiperVoice
from openwakeword.model import Model
from faster_whisper import WhisperModel

from rag_engine import search_in_folder, should_search_files
from vision_engine import is_vision_request, capture_frame, query_vlm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Common hallucinated phrases faster-whisper sometimes produces on silence,
# in both English and Italian (seen with language="it" on near-silent audio).
HALLUCINATION_PHRASES = ("Thank you", "Thanks for watching", "Alla prossima")


def notify_ui(role: str, text: str):
    """Pushes a message to the local web UI (server.py), if it's running.
    Best-effort: the assistant keeps working even if the UI is closed."""
    try:
        requests.post(f"{UI_SERVER_URL}/send_message", json={"role": role, "text": text}, timeout=0.5)
    except requests.RequestException:
        pass


def main():
    logger.info("Initializing voice pipeline...")
    notify_ui("sys", "Avvio assistente: caricamento modelli pesanti in corso (Whisper, VAD)...")

    # 1. Load wake word model (official download helper handles auth/URL internally)
    openwakeword.utils.download_models(model_names=[WAKE_WORD_MODEL])
    oww_model = Model(wakeword_models=[WAKE_WORD_MODEL], inference_framework="onnx")

    # 2. Load Silero VAD (via Torch Hub)
    vad_model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    vad_model.eval()

    # 3. Load faster-whisper (CUDA acceleration)
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="float16")
    logger.info("Models successfully loaded on GPU.")
    notify_ui("sys", "Modelli vocali allocati su GPU. Caricamento voce TTS...")

    # 4. Load Piper TTS (Italian voice)
    piper_voice = PiperVoice.load(PIPER_MODEL_PATH, config_path=PIPER_CONFIG_PATH)

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1280       # wake word frame size (80ms), no strict size constraint
    VAD_CHUNK = 512    # Silero VAD at 16kHz requires EXACTLY 512 samples per frame
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    silence_threshold = 1.5  # seconds of silence that end a recorded command
    vad_chunks_per_second = RATE / VAD_CHUNK
    max_silent_chunks = int(silence_threshold * vad_chunks_per_second)
    max_initial_wait = int(5.0 * vad_chunks_per_second)  # silence before giving up on standby

    logger.info("Listening for the wake word... (Ctrl+C to stop)")
    notify_ui("sys", "Sistema pronto. In attesa della parola di attivazione...")

    conversation_history = [{
        "role": "system",
        "content": (
            "Sei un assistente vocale. Rispondi in italiano con al massimo 2-3 frasi brevi, "
            "in prosa, senza elenchi, markdown o emoji: la risposta verrà letta ad alta voce."
        ),
    }]
    conversation_mode = False

    def speak(text_to_speak: str):
        """Synthesizes speech with Piper and plays it, blocking until finished
        (prevents the microphone from picking up the assistant's own voice)."""
        tts_buffer = io.BytesIO()
        with wave.open(tts_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(piper_voice.config.sample_rate)
            for audio_chunk in piper_voice.synthesize(text_to_speak):
                wav_file.writeframes(audio_chunk.audio_int16_bytes)

        tts_buffer.seek(0)
        with wave.open(tts_buffer, "rb") as wf:
            stream_out = audio.open(
                format=audio.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
            )
            data = wf.readframes(1024)
            while data:
                stream_out.write(data)
                data = wf.readframes(1024)
            stream_out.stop_stream()
            stream_out.close()

    try:
        while True:
            # PHASE 1: wake word detection (standby only)
            if not conversation_mode:
                audio_data = stream.read(CHUNK, exception_on_overflow=False)
                numpy_data = np.frombuffer(audio_data, dtype=np.int16)

                prediction = oww_model.predict(numpy_data)
                score = list(prediction.values())[0]

                if score > WAKE_WORD_THRESHOLD:
                    logger.info("Wake word detected. Conversation mode ACTIVE.")
                    oww_model.reset()
                    conversation_mode = True
                    stream.read(stream.get_read_available(), exception_on_overflow=False)

            # PHASE 2: VAD-controlled recording
            if conversation_mode:
                recorded_frames = []
                silent_chunks = 0
                has_spoken = False

                logger.info("Listening...")
                notify_ui("sys", "Ascolto in corso...")

                while True:
                    rec_data = stream.read(VAD_CHUNK, exception_on_overflow=False)
                    recorded_frames.append(rec_data)

                    chunk_np = np.frombuffer(rec_data, dtype=np.int16).astype(np.float32) / 32768.0
                    chunk_tensor = torch.from_numpy(chunk_np)
                    speech_prob = vad_model(chunk_tensor, RATE).item()

                    if speech_prob > 0.3:
                        silent_chunks = 0
                        has_spoken = True
                    else:
                        silent_chunks += 1

                    if has_spoken and silent_chunks > max_silent_chunks:
                        break
                    if not has_spoken and silent_chunks > max_initial_wait:
                        logger.info("Inactivity timeout. Returning to standby.")
                        notify_ui("sys", "Nessun comando vocale. Torno in standby.")
                        conversation_mode = False
                        break

                if not conversation_mode:
                    continue

                # PHASE 3: speech-to-text
                logger.info("Transcribing...")
                notify_ui("sys", "Trascrizione in corso...")
                full_audio = b"".join(recorded_frames)
                audio_np = np.frombuffer(full_audio, dtype=np.int16).astype(np.float32) / 32768.0

                segments, _info = whisper_model.transcribe(
                    audio_np,
                    beam_size=5,
                    language="it",
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                )
                user_text = "".join(segment.text for segment in segments).strip()

                if not user_text or len(user_text) <= 2 or any(p in user_text for p in HALLUCINATION_PHRASES):
                    continue

                logger.info(f"USER: {user_text}")
                notify_ui("user", user_text)

                wants_file_search, folder_alias = should_search_files(user_text)
                wants_vision = is_vision_request(user_text)

                try:
                    if wants_vision:
                        logger.info("Vision request detected, taking photo...")
                        notify_ui("sys", "Acquisizione e analisi visiva in corso...")
                        photo_path = capture_frame()
                        llm_response = (
                            query_vlm(user_text, photo_path) if photo_path
                            else "Non riesco ad accedere alla camera."
                        )
                        conversation_history.append({"role": "user", "content": user_text})
                        conversation_history.append({"role": "assistant", "content": llm_response})

                    elif wants_file_search:
                        logger.info(f"Searching in files: '{folder_alias}'...")
                        notify_ui("sys", f"Ricerca nei documenti: {folder_alias}...")
                        context = search_in_folder(folder_alias, user_text)

                        system_prompt = (
                            "Rispondi in italiano in modo molto conciso (massimo 2 frasi brevi). "
                            f"Usa ESCLUSIVAMENTE questo contesto estratto dai file dell'utente:\n{context}"
                        )
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_text},
                        ]
                        response = ollama.chat(model=LLM_MODEL, messages=messages)
                        llm_response = response["message"]["content"]

                        # Minimal follow-up memory: question + answer only, not the
                        # full retrieved context (too long to keep around).
                        conversation_history.append({"role": "user", "content": user_text})
                        conversation_history.append({"role": "assistant", "content": llm_response})

                    else:
                        logger.info("Processing standard response...")
                        conversation_history.append({"role": "user", "content": user_text})
                        response = ollama.chat(model=LLM_MODEL, messages=conversation_history)
                        llm_response = response["message"]["content"]
                        conversation_history.append({"role": "assistant", "content": llm_response})

                    logger.info(f"ASSISTANT: {llm_response}")
                    notify_ui("ai", llm_response)
                    speak(llm_response)

                except Exception as e:
                    logger.error(f"Ollama / TTS / VLM error: {e}")

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":
    main()
