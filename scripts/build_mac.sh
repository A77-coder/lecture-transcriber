#!/bin/bash
# macOS版 スタンドアロンアプリ (.app / .dmg) ビルドスクリプト
# LectureTranscriber_mac.spec を使用し、Retina対応・マイク権限(Info.plist)込みで生成する

set -e
cd "$(dirname "$0")/.."

echo "=== Lecture Transcriber macOS アプリビルド開始 ==="

# Python 3.10〜3.12 を優先探索
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "エラー: python3 が見つかりません。Homebrew (brew install python@3.12) 等でインストールしてください。"
    exit 1
fi

echo "[1/4] 仮想環境と依存パッケージを準備中 ($PYTHON_BIN)..."
if [ ! -f ".venv_mac/bin/python" ]; then
    "$PYTHON_BIN" -m venv .venv_mac
fi
.venv_mac/bin/python -m pip install --upgrade pip
.venv_mac/bin/python -m pip install -r requirements.txt pyinstaller

echo "[2/4] LectureTranscriber.app をビルド中..."
rm -rf build dist/LectureTranscriber dist/LectureTranscriber.app
.venv_mac/bin/pyinstaller --noconfirm --clean LectureTranscriber_mac.spec

echo "[3/4] アドホック署名（Gatekeeper/権限安定化）を適用中..."
if command -v codesign >/dev/null 2>&1; then
    codesign --force --deep --sign - "dist/LectureTranscriber.app" || true
fi

echo "[4/4] 配布用DMGイメージを作成中..."
if command -v hdiutil >/dev/null 2>&1; then
    DMG_STAGE="dist/dmg_stage"
    rm -rf "$DMG_STAGE" "dist/LectureTranscriber-macOS.dmg"
    mkdir -p "$DMG_STAGE"
    cp -R "dist/LectureTranscriber.app" "$DMG_STAGE/"
    # ドラッグ＆ドロップで簡単にインストールできるよう Applications へのリンクを配置
    ln -s /Applications "$DMG_STAGE/Applications"

    hdiutil create -volname "Lecture Transcriber" \
        -srcfolder "$DMG_STAGE" \
        -ov -format UDZO \
        "dist/LectureTranscriber-macOS.dmg"
    rm -rf "$DMG_STAGE"
    echo "=== 完了！ ==="
    echo "生成ファイル:"
    echo "  - dist/LectureTranscriber.app"
    echo "  - dist/LectureTranscriber-macOS.dmg"
else
    echo "=== 完了: dist/LectureTranscriber.app ==="
fi
