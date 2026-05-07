@echo off
cd /d "%~dp0"
echo Starting Gene Expression Predictor...
start "" http://localhost:8000
venv\Scripts\python.exe web\main.py
pause
