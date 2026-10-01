import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model


def main():
    # One-time download of the pre-trained model (cached locally after the first run)
    print("[SYSTEM] Checking / downloading wake word model...")
    openwakeword.utils.download_models(model_names=["alexa"])

    oww_model = Model(wakeword_models=["alexa"], inference_framework="onnx")

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1280
    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("[SYSTEM] Listening for 'Alexa'... (Press Ctrl+C to stop)")

    try:
        while True:
            audio_data = stream.read(CHUNK, exception_on_overflow=False)
            numpy_data = np.frombuffer(audio_data, dtype=np.int16)

            prediction = oww_model.predict(numpy_data)

            for mdl_name, score in prediction.items():
                if score > 0.5:
                    print(f"\n[!] Wake word detected! (Confidence: {score:.2f})")
                    oww_model.reset()

    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutting down...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":
    main()