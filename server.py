from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse

app = FastAPI()
connected_clients = set()

html_content = """
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Assistente Vocale</title>
    <style>
        body { font-family: sans-serif; background-color: #121212; color: #ffffff; padding: 2rem; max-width: 800px; margin: 0 auto; }
        #status { padding: 10px; margin-bottom: 20px; border-radius: 5px; background-color: #1e1e1e; font-weight: bold; }
        #chat { display: flex; flex-direction: column; gap: 10px; }
        .msg { padding: 10px; border-radius: 8px; max-width: 80%; }
        .user { background-color: #005c4b; align-self: flex-end; }
        .ai { background-color: #2d2d2d; align-self: flex-start; }
        .sys { color: #888; font-size: 0.9em; font-style: italic; align-self: center; }
    </style>
</head>
<body>
    <h2>Serena AI</h2>
    <div id="status">Stato: Disconnesso</div>
    <div id="chat"></div>

    <script>
        const ws = new WebSocket("ws://localhost:8000/ws");
        const chat = document.getElementById("chat");
        const status = document.getElementById("status");

        ws.onopen = () => status.innerText = "Stato: Connesso";
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            const div = document.createElement("div");
            div.classList.add("msg", data.role);
            div.innerText = data.text;
            chat.appendChild(div);
            window.scrollTo(0, document.body.scrollHeight);
        };
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        connected_clients.remove(websocket)

# Endpoint che riceve i messaggi dal tuo script Python e li manda alla UI
@app.post("/send_message")
async def send_message(request: Request):
    data = await request.json()
    for client in connected_clients.copy():
        try:
            await client.send_json(data)
        except:
            connected_clients.remove(client)
    return {"status": "ok"}