@echo off
REM Ativa o venv e roda o ingest a partir da raiz do projeto
cd /d C:\Users\mvl_t\PycharmProjects\pi4-weather

REM Executa direto o python do venv
".venv\Scripts\python.exe" -m src.ingest 1>> logs\ingest_ok.log 2>> logs\ingest_err.log
