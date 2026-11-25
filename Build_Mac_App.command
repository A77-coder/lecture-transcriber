#!/bin/bash
# macOS Finderからダブルクリックで LectureTranscriber.app / .dmg を作成するランチャー
cd "$(dirname "$0")"
bash scripts/build_mac.sh
echo ""
echo "ビルドが完了しました。dist フォルダを開きます..."
open dist
read -p "Enterキーを押すとウィンドウを閉じます..."
