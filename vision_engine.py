import cv2
import base64
import requests
import os

URL = "http://192.168.1.113:8080/video"
TRIGGER_VISIVI = ["guarda", "vedi", "dov'è", "cosa c'è", "mostra"]

def cattura_frame():
    print("[VISION] Cattura frame da Ip Webcam in corso...")
    cap = cv2.VideoCapture(URL)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)
        return temp_path
    
    print("[VISION] Errore: Impossibile leggere il flusso video.")
    return None

def is_vision_request(testo):
    """Controlla se l'utente ha usato parole chiave visive."""
    testo_lower = testo.lower()
    return any(trigger in testo_lower for trigger in TRIGGER_VISIVI)

def interroga_vlm(prompt, image_path):
    print("[VISION] Invio immagine e prompt al VLM locale...")
    with open(image_path, "rb") as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
        
    payload = {
        "model": "qwen2.5vl:3b",
        "prompt": "Descrivi in italiano cosa vedi in questa foto.",
        "images": [img_b64],
        "stream": False,
        "keep_alive": 0,
        "options": {
            "temperature": 0.4
        }
    }
    
    try:
        response = requests.post("http://127.0.0.1:11434/api/generate", json=payload)
        response.raise_for_status()
        
        # Elimina la foto temporanea per non accumulare spazzatura
        if os.path.exists(image_path):
            os.remove(image_path)
            
        return response.json().get("response", "Errore nella risposta del modello visivo.")
    except Exception as e:
        return f"Errore di connessione a Ollama: {e}"