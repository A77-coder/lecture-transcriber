@echo off
chcp 65001 > nul
REM Windows版 実行ファイル (.exe) ビルドスクリプト (PyInstaller使用)

cd /d "%~dp0\.."
echo [1/3] ビルド用パッケージ (PyInstaller) を確認中...
.venv\Scripts\python.exe -m pip install pyinstaller

echo [2/3] Windows版アプリをビルド中...
.venv\Scripts\pyinstaller.exe ^
    --noconfirm ^
    --clean ^
    --windowed ^
    --name "LectureTranscriber" ^
    --collect-all faster_whisper ^
    --collect-all ctranslate2 ^
    --hidden-import pyaudiowpatch ^
    main.py

echo [3/3] ビルド完了: dist\LectureTranscriber\LectureTranscriber.exe
pause
