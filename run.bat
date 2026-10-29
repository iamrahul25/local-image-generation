@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo App environment not found. Run setup.bat first.
    exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
