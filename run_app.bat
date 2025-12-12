@echo off
REM Launcher for CageMetrics
REM Activates plethapp conda environment and runs the app

echo ============================================================
echo  CageMetrics - Behavioral Analysis
echo ============================================================
echo.

REM Initialize conda for this shell session
call C:\Users\rphil2\AppData\Local\miniforge3\Scripts\activate.bat

REM Activate the plethapp environment
call conda activate plethapp

REM Change to project directory
cd /d "%~dp0"

echo Starting application...
echo.

REM Run the app
python run_debug.py

REM Keep window open if there was an error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%
    pause
)
