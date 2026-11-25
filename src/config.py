# -*- coding: utf-8 -*-
"""アプリケーション設定・定数・プラットフォーム判定モジュール"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path

# 現在のOSを判定する定数
OS_NAME = platform.system()  # "Windows", "Darwin" (macOS), "Linux"
IS_WINDOWS = OS_NAME == "Windows"
IS_MAC = OS_NAME == "Darwin"

# Whisperモデル一覧（UI表示名とモデルIDの対応）
AVAILABLE_MODELS: list[tuple[str, str]] = [
    ("tiny (最速・軽量 / テスト向け)", "tiny"),
    ("base (高速・標準精度)", "base"),
    ("small (バランス型・講義におすすめ)", "small"),
    ("medium (高精度・やや重い)", "medium"),
    ("large-v3-turbo (最高精度・高速 / GPU推奨)", "large-v3-turbo"),
]

# 対応言語一覧
AVAILABLE_LANGUAGES: list[tuple[str, str]] = [
    ("日本語 (ja)", "ja"),
    ("英語 (en)", "en"),
    ("自動検出 (auto)", "auto"),
]

# 音声処理の固定パラメータ
TARGET_SAMPLE_RATE = 16000  # Whisperが期待するサンプルレート (16kHz)

# 設定ファイルの保存先ディレクトリ
CONFIG_DIR = Path.home() / ".lecture_transcriber"
CONFIG_FILE = CONFIG_DIR / "config.json"


@dataclass
class AppConfig:
    """ユーザー設定を保持するデータクラス"""

    model_size: str = "small"  # 使用するWhisperモデルサイズ
    language: str = "ja"  # 文字起こし言語 ("ja", "en", "auto")
    device_preference: str = "auto"  # 推論デバイス ("auto", "cuda", "cpu")
    custom_prompt: str = ""  # 講義テーマや専門用語のヒント（手動入力用）
    glossary_files: list[str] = field(default_factory=list)  # 読み込み済みの単語帳MD/TXTパス一覧
    glossary_section: str = ""  # 単語帳の絞り込みセクション（空文字で全セクション自動追従）
    silence_threshold: float = 0.003  # 無音判定のRMS音量閾値（小さめの動画音量にも対応）
    silence_duration_sec: float = 0.75  # 発話区切りとみなす無音継続時間（秒）
    max_segment_sec: float = 18.0  # 1セグメントの最大長（秒）
    partial_interval_sec: float = 1.6  # 暫定テキスト（リアルタイム仮表示）の更新間隔（秒）
    auto_scroll: bool = True  # 文字起こしログの自動スクロール有効/無効
    overlay_font_size: int = 20  # フローティング字幕のフォントサイズ
    overlay_opacity: float = 0.82  # フローティング字幕の背景不透明度 (0.3〜1.0)
    last_device_id: str = ""  # 前回選択していた音声入力デバイスの識別子

    @classmethod
    def load(cls) -> "AppConfig":
        """設定ファイルから設定を読み込む（存在しない場合はデフォルト値）"""
        try:
            if CONFIG_FILE.exists():
                data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                # 定義済みのフィールドのみ抽出して安全に読み込む
                valid_keys = {f for f in cls.__dataclass_fields__}
                filtered = {k: v for k, v in data.items() if k in valid_keys}
                return cls(**filtered)
        except Exception:
            # 破損時はデフォルト設定にフォールバック
            pass
        return cls()

    def save(self) -> None:
        """現在の設定をJSONファイルに保存する"""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
