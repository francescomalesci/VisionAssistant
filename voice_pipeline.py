import os

# dll_dir = r"C:\Users\franc\Downloads\Faster-Whisper-XXL_r245.4_windows\Faster-Whisper-XXL\_xxl_data\torch\lib"
dll_dir = r"D:\Francy\Download\Faster-Whisper-XXL_r245.4_windows\Faster-Whisper-XXL\_xxl_data\torch\lib"
os.add_dll_directory(dll_dir)
os.environ["PATH"] = dll_dir + os.pathsep + os.environ["PATH"]

import io
import wave
import pyaudio
import numpy as np
import torch
import openwakeword
import ollama
import asyncio
import requests
from piper import PiperVoice
from openwakeword.model import Model
from faster_whisper import WhisperModel
from rag_engine import search_in_folder, should_search_files
from vision_engine import is_vision_request, cattura_frame, interroga_vlm


def main():
    print("[SYSTEM] Initializing voice pipeline...")

    #0. Invia primo avviso caricamento
    def aggiorna_ui(role, text):
        try:
            requests.post("http://127.0.0.1:8000/send_message", json={"role": role, "text": text}, timeout=0.5)
        except:
            pass

    aggiorna_ui("sys", "Avvio assistente: caricamento modelli pesanti in corso (Whisper, VAD)...")

    # 1. Load Wake Word (official download, handles auth/URL internally)
    openwakeword.utils.download_models(model_names=["alexa"])
    oww_model = Model(wakeword_models=["alexa"], inference_framework="onnx")

    # 2. Load Silero VAD (via Torch Hub)
    vad_model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        trust_repo=True
    )
    vad_model.eval()

    # 3. Load Faster-Whisper (CUDA acceleration)
    whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
    print("[SYSTEM] Models successfully loaded on GPU.")

    aggiorna_ui("sys", "Modelli visivi e uditivi allocati su GPU. Caricamento voce TTS...")

    # 4. Load Piper TTS
    print("[SYSTEM] Loading Piper TTS model...")
    # WARNING: Update this filename to an English voice model (e.g., en_US-lessac-medium.onnx)
    piper_voice = PiperVoice.load(
        "it_IT-serena-medium.onnx",
        config_path="it_IT-serena-medium.onnx.json"
    )

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1280       # Frames for wake word (80ms, no size constraints)
    VAD_CHUNK = 512    # Silero VAD at 16kHz requires EXACTLY 512 samples per frame
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    # Logical parameters for recording (calculated on VAD frame size)
    silence_threshold = 1.5  # Stops recording after 1.5 seconds of silence
    vad_chunks_per_second = RATE / VAD_CHUNK
    max_silent_chunks = int(silence_threshold * vad_chunks_per_second)

    print("\n[SYSTEM] Listening for 'Alexa'... (Press Ctrl+C to stop)")
    aggiorna_ui("sys", "Sistema pronto. In attesa della parola di attivazione 'Alexa'...")
    conversation_history = []
    conversation_mode = False

    # 5 seconds of initial silence timeout before exiting conversation mode
    max_initial_wait = int(5.0 * vad_chunks_per_second)

    def speak(text_to_speak: str):
        """Synthesizes with Piper and plays audio, blocking until finished."""
        tts_audio_buffer = io.BytesIO()
        with wave.open(tts_audio_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(piper_voice.config.sample_rate)
            for audio_chunk in piper_voice.synthesize(text_to_speak):
                wav_file.writeframes(audio_chunk.audio_int16_bytes)

        tts_audio_buffer.seek(0)
        with wave.open(tts_audio_buffer, 'rb') as wf:
            stream_out = audio.open(
                format=audio.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )
            data = wf.readframes(1024)
            while data:
                stream_out.write(data)
                data = wf.readframes(1024)
            stream_out.stop_stream()
            stream_out.close()

    def aggiorna_ui(role, text):
        """Invia i log all'interfaccia web senza bloccare il codice."""
        try:
            requests.post("http://127.0.0.1:8000/send_message", json={"role": role, "text": text}, timeout=0.5)
        except:
            pass # Se l'interfaccia è chiusa, ignora l'errore e continua a funzionare

    try:
        while True:
            # PHASE 1: Wake Word Detection (only runs in standby)
            if not conversation_mode:
                audio_data = stream.read(CHUNK, exception_on_overflow=False)
                numpy_data = np.frombuffer(audio_data, dtype=np.int16)

                prediction = oww_model.predict(numpy_data)
                score = list(prediction.values())[0]

                if score > 0.75:
                    print("\n[!] Wake word detected. Conversation mode ACTIVE.")
                    oww_model.reset()
                    conversation_mode = True
                    stream.read(stream.get_read_available(), exception_on_overflow=False)

            # PHASE 2: Continuous Recording
            if conversation_mode:
                recorded_frames = []
                silent_chunks = 0
                has_spoken = False

                print("[SYSTEM] Listening...")
                aggiorna_ui("sys", "Ascolto in corso...")

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
                        print("[SYSTEM] Inactivity timeout. Returning to standby...")
                        aggiorna_ui("sys", "Nessun comando vocale. Torno in standby.")
                        conversation_mode = False
                        break

                if not conversation_mode:
                    continue

                # PHASE 3: Whisper STT Processing
                print("[SYSTEM] Processing transcription...")
                aggiorna_ui("sys", "Trascrizione in corso...")
                full_audio = b''.join(recorded_frames)
                audio_np = np.frombuffer(full_audio, dtype=np.int16).astype(np.float32) / 32768.0

                segments, info = whisper_model.transcribe(
                    audio_np,
                    beam_size=5,
                    language="it",
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6
                )
                user_text = "".join([segment.text for segment in segments]).strip()

                # Filter common Whisper hallucinations in English
                if not user_text or len(user_text) <= 2 or "Thank you" in user_text or "Thanks for watching" in user_text:
                    continue

                print(f"\n=> USER: {user_text}")
                aggiorna_ui("user", user_text)

                search_files, folder_alias = should_search_files(user_text)
                vision_request = is_vision_request(user_text)

                try:
                    if vision_request:
                        print("[SYSTEM] Vision request detected, taking photo...")
                        aggiorna_ui("sys", "Acquisizione e analisi visiva in corso...") # <-- 2. Mostra lo stato visivo
                        photo_path = cattura_frame()
                        if photo_path:
                            llm_response = interroga_vlm(user_text, photo_path)
                        else:
                            llm_response = "Non riesco ad accedere alla camera."
                        
                        conversation_history.append({'role': 'user', 'content': user_text})
                        conversation_history.append({'role': 'assistant', 'content': llm_response})

                    elif search_files:
                        print(f"[SYSTEM] Searching in files: '{folder_alias}'...")
                        aggiorna_ui("sys", f"Ricerca nei documenti: {folder_alias}...") # <-- 3. Mostra lo stato di ricerca RAG
                        context = search_in_folder(folder_alias, user_text)

                        system_prompt = (
                            "Answer concisely in English (maximum 2 sentences). "
                            f"Use EXCLUSIVELY this context extracted from the user's files:\n{context}"
                        )
                        messages = [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': user_text}
                        ]
                        response = ollama.chat(model='qwen2.5:7b', messages=messages)
                        llm_response = response['message']['content']

                        conversation_history.append({'role': 'user', 'content': user_text})
                        conversation_history.append({'role': 'assistant', 'content': llm_response})

                    else:
                        print("[SYSTEM] Processing standard response...")
                        conversation_history.append({'role': 'user', 'content': user_text})
                        response = ollama.chat(model='qwen2.5:7b', messages=conversation_history)
                        llm_response = response['message']['content']
                        conversation_history.append({'role': 'assistant', 'content': llm_response})

                    print(f"=> ASSISTANT: {llm_response}\n")
                    aggiorna_ui("ai", llm_response) # <-- 4. Mostra la risposta dell'AI
                    speak(llm_response)

                except Exception as e:
                    print(f"[ERROR] Ollama connection / TTS / VLM Error: {e}\n")

    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutting down application...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":
    main()