@echo off
REM Debug launcher for CageMetrics
REM Shows console output for debugging - keeps window open after exit

echo ============================================================
echo  CageMetrics - DEBUG MODE
echo ============================================================
echo.

REM Initialize conda for this shell session
call C:\Users\rphil2\AppData\Local\miniforge3\Scripts\activate.bat

REM Activate the plethapp environment
call conda activate plethapp

REM Change to project directory
cd /d "%~dp0"

echo Starting application in debug mode...
echo Console output will be shown below:
echo ------------------------------------------------------------
echo.

REM Run the app
python run_debug.py

echo.
echo ------------------------------------------------------------
echo Application exited with code %ERRORLEVEL%
echo.
pause
