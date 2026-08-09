@echo off
echo ============================================
echo EVENT SNAP BACKEND - Starting on Port 8081
echo ============================================
echo.

cd /d "%~dp0"

echo Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)
echo.

echo Starting server...
echo Server will run on: http://localhost:8081
echo API endpoint: http://localhost:8081/api/
echo.
echo Press CTRL+C to stop
echo.

uvicorn server:app --host 127.0.0.1 --port 8081 --log-level info

pause
