@echo off
chcp 65001 > nul
REM Windows用 起動スクリプト

cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo 仮想環境が見つかりません。セットアップを実行します...
    python -m venv .venv
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements-win.txt
)

start "" ".venv\Scripts\pythonw.exe" main.py
