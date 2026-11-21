@echo off
setlocal
cd /d "%~dp0"
echo ============================================
echo   Smart Academic Help - one-time setup
echo ============================================

where py >nul 2>nul
if %errorlevel%==0 (set "PY=py -3") else (set "PY=python")
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [X] Python 3.10 or newer not found.
    echo     Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/5] Creating virtual environment .venv ...
    %PY% -m venv .venv || (echo [X] Could not create .venv & pause & exit /b 1)
) else (
    echo [1/5] Virtual environment already exists.
)
set "VPY=.venv\Scripts\python.exe"

echo [2/5] Installing packages (first time takes 5-10 minutes) ...
"%VPY%" -m pip install --upgrade pip -q
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (echo [X] Package install failed - check your internet and run setup.bat again. & pause & exit /b 1)

echo [3/5] NLTK data ...
set "NEED_NLTK=0"
if not exist "nltk_data\corpora\wordnet" set "NEED_NLTK=1"
if not exist "nltk_data\taggers\averaged_perceptron_tagger_eng" set "NEED_NLTK=1"
if "%NEED_NLTK%"=="1" (
    "%VPY%" -m sah.setup_nltk
) else (
    echo       already included in the project.
)

echo.
echo [4/5] Optional: transformer embeddings for better search (~300 MB download).
choice /c YN /t 20 /d N /m "      Install now? Auto-answers N in 20 seconds"
if errorlevel 2 (
    echo       Skipped. Install later with: .venv\Scripts\python -m pip install sentence-transformers faiss-cpu
) else (
    "%VPY%" -m pip install sentence-transformers faiss-cpu
)

if not exist ".env" copy ".env.example" ".env" >nul

echo [5/5] Running tests ...
"%VPY%" -m pytest -q -p no:cacheprovider
echo.
echo ============================================
echo   Setup finished.
echo   1. Paste your Groq key into .env  (GROQ_API_KEY=...)
echo      Free key, no card: https://console.groq.com/keys
echo   2. Double-click run_app.bat
echo ============================================
findstr /r /c:"^GROQ_API_KEY=..*" .env >nul || start "" notepad .env
pause
