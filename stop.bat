@echo off
REM ITAC AI - stop the app. Double-click this file.
setlocal
cd /d "%~dp0"
set "PATH=%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin;%PATH%"

echo Stopping ITAC AI...
docker compose down
echo Done.
timeout /t 3 /nobreak >nul
