@echo off
REM One-time setup: creates a virtual environment in .venv and installs dependencies.
cd /d "%~dp0"
echo Installing VS Code project settings into .vscode ...
if not exist .vscode mkdir .vscode
xcopy /y /q vscode_config\*.json .vscode\ >nul || goto :error
if not exist .venv (
    echo Creating virtual environment in .venv ...
    python -m venv .venv || goto :error
)
echo Installing dependencies ...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :error
echo.
echo Done. In VS Code: Ctrl+Shift+P ^> "Python: Select Interpreter" ^> .venv
goto :eof

:error
echo Setup failed. Make sure Python 3.10+ is installed and on PATH.
exit /b 1
