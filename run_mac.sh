#!/bin/bash
# macOS用 起動スクリプト

set -e
cd "$(dirname "$0")"

if [ ! -f ".venv/bin/python" ]; then
    echo "仮想環境が見つかりません。セットアップを実行します..."
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements-mac.txt
fi

.venv/bin/python main.py
