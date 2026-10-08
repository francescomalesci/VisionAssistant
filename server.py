"""
Minimal local web UI: shows a live chat log of the voice assistant's
conversation (user transcript, system status, assistant replies) over a
WebSocket. The voice pipeline pushes messages to it via POST /send_message.

This is a local debugging/demo UI, not intended to be exposed outside the
local machine: /send_message has no authentication.
"""
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

app = FastAPI()
connected_clients: set[WebSocket] = set()

HTML_PAGE = """
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Assistant</title>
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
    <h2>Assistente AI</h2>
    <div id="status">Status: disconnected</div>
    <div id="chat"></div>

    <script>
        const ws = new WebSocket("ws://localhost:8000/ws");
        const chat = document.getElementById("chat");
        const status = document.getElementById("status");

        ws.onopen = () => status.innerText = "Status: connected";

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
async def index():
    return HTMLResponse(HTML_PAGE)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # kept alive; client doesn't send anything meaningful
    except WebSocketDisconnect:
        connected_clients.discard(websocket)


@app.post("/send_message")
async def send_message(request: Request):
    """Receives a message from the voice pipeline and broadcasts it to every
    connected browser tab."""
    data = await request.json()
    for client in connected_clients.copy():
        try:
            await client.send_json(data)
        except Exception:
            connected_clients.discard(client)
    return {"status": "ok"}
