@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.bat first.
    pause
    exit /b 1
)
echo REST API docs: http://localhost:8000/docs  (close this window to stop)
".venv\Scripts\python.exe" -m uvicorn api:app --host 0.0.0.0 --port 8000
pause
