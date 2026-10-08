@echo off
cd /d "%~dp0"

echo Verifica che Ollama sia in esecuzione prima di continuare...

echo Avvio del server web...
start cmd /k "venv\Scripts\activate && uvicorn server:app --host 0.0.0.0 --port 8000"

:: Attende 2 secondi per garantire che FastAPI sia in ascolto
timeout /t 2 /nobreak >nul

:: Apre automaticamente l'interfaccia nel browser predefinito
start http://localhost:8000

echo Avvio della pipeline vocale...
call venv\Scripts\activate
python voice_pipeline.py
