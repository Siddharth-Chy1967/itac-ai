@echo off
REM ITAC AI - one-click launch. Double-click this file.
REM Starts Docker Desktop if needed, brings up the containers, opens the app.
setlocal
cd /d "%~dp0"

REM Make sure Docker's tools (docker + credential helper) are on PATH even if this
REM window predates the install.
set "PATH=%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin;%PATH%"

echo Checking Docker engine...
docker info >nul 2>&1
if not %errorlevel%==0 (
  echo Docker isn't running yet - starting Docker Desktop...
  start "" "%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe"
  echo Waiting for the engine ^(first start can take a minute^)...
  :waitloop
  timeout /t 4 /nobreak >nul
  docker info >nul 2>&1
  if not %errorlevel%==0 goto waitloop
)

echo Starting ITAC AI...
docker compose up -d
if not %errorlevel%==0 (
  echo.
  echo Failed to start. See the errors above.
  pause
  exit /b 1
)

start "" http://localhost:8080
echo.
echo ================================================
echo   ITAC AI is running at http://localhost:8080
echo ================================================
timeout /t 4 /nobreak >nul
