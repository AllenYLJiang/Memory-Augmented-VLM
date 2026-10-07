@echo off
cd /d "%~dp0"
python -B run_local.py %*
exit /b %errorlevel%
