@echo off
setlocal enableextensions
cd /d "%~dp0"
echo ============================================================
echo  Open Generative AI - Streamlit Studio : one-time setup
echo ============================================================

where python >nul 2>nul || (echo [ERROR] Python 3.10+ is required: https://www.python.org/downloads/ & exit /b 1)
where git >nul 2>nul || (echo [ERROR] Git is required: https://git-scm.com/download/win & exit /b 1)
where node >nul 2>nul || (echo [ERROR] Node.js 18+ is required to read the repo model catalog: https://nodejs.org & exit /b 1)

echo.
echo [1/6] Installing uv (fast Python package manager)...
python -m pip install --user --quiet uv || exit /b 1

echo.
echo [2/6] Creating the Streamlit app environment (.venv, Python 3.12)...
if not exist ".venv\Scripts\python.exe" python -m uv venv .venv --python 3.12 || exit /b 1
python -m uv pip install --python .venv\Scripts\python.exe -r requirements.txt || exit /b 1

echo.
echo [3/6] Cloning Open-Generative-AI and exporting its model catalog...
if not exist "Open-Generative-AI\package.json" git clone --recurse-submodules https://github.com/Anil-matcha/Open-Generative-AI.git || exit /b 1
node --no-warnings scripts\export_catalog.mjs || exit /b 1

echo.
echo [4/6] Installing sd.cpp (CUDA) + default local image model (Z-Image Turbo, ~5.2 GB)...
.venv\Scripts\python.exe scripts\setup_local.py --all || exit /b 1

echo.
echo [5/6] Installing Wan2GP (local video engine, Python 3.11 + PyTorch CUDA, ~6 GB)...
if not exist "engines\Wan2GP\wgp.py" git clone --depth 1 https://github.com/deepbeepmeep/Wan2GP.git engines\Wan2GP || exit /b 1
pushd engines\Wan2GP
if not exist ".venv\Scripts\python.exe" python -m uv venv .venv --python 3.11 || exit /b 1
python -m uv pip install --python .venv\Scripts\python.exe torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130 || exit /b 1
python -m uv pip install --python .venv\Scripts\python.exe -r requirements.txt --index-strategy unsafe-best-match || exit /b 1
python -m uv pip install --python .venv\Scripts\python.exe -U "triton-windows>=3.6,<3.7"
popd

echo.
echo [6/6] Local LLM (Ollama)...
where ollama >nul 2>nul
if errorlevel 1 (
    echo Ollama not found. Install it with:  winget install Ollama.Ollama   then run:  ollama pull qwen3:4b-instruct
) else (
    ollama pull qwen3:4b-instruct
)

echo.
echo Setup complete. Start the app with:  run.bat
echo The Wan 2.1 1.3B video weights (~8 GB) download automatically on your first local video generation.
endlocal
