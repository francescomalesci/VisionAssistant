import ollama

# Legge esplicitamente il file in binario
with open("test_droidcam.jpg", "rb") as img_file:
    img_bytes = img_file.read()

response = ollama.generate(
    model="qwen2.5vl:3b",
    prompt="What is in this image? Answer with one short sentence in Italian.",
    images=[img_bytes],
    options={
        "temperature": 0.2,       # Riduce le allucinazioni
        "repeat_penalty": 1.2,    # Penalizza fortemente l'uso ripetuto degli stessi token
        "num_predict": 100        # Taglia forzatamente la risposta prima che si formi un loop infinito
    }
)

print(response["response"])