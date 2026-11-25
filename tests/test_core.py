# -*- coding: utf-8 -*-
"""コア機能（音声リサンプリング・エクスポーター・設定管理）の単体テスト"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from src.audio_capture import resample_to_16k_mono
from src.config import AppConfig, TARGET_SAMPLE_RATE
from src.file_transcriber import iter_media_audio_chunks
from src.glossary import (
    GlossaryDictionary,
    extract_keywords_from_file,
    extract_keywords_from_markdown,
)
from src.transcriber import _clean_text, _join_segments, _normalize_audio
from src.exporter import (
    TranscriptSegment,
    export_segments,
    format_hhmmss,
    format_srt_time,
    segments_to_markdown,
    segments_to_srt,
    segments_to_txt,
)


def test_resample_to_16k_mono_stereo_48k() -> None:
    """48kHz ステレオ音声が 16kHz モノラルに正しく変換されることを検証する"""
    duration_sec = 0.5
    src_rate = 48000
    n_frames = int(src_rate * duration_sec)
    # 440Hzサイン波のステレオデータを作成
    t = np.linspace(0, duration_sec, n_frames, endpoint=False, dtype=np.float32)
    wave = 0.5 * np.sin(2 * np.pi * 440 * t)
    stereo = np.stack([wave, wave], axis=1)

    out = resample_to_16k_mono(stereo, src_rate=src_rate, channels=2)
    assert out.ndim == 1
    assert out.dtype == np.float32
    expected_len = int(TARGET_SAMPLE_RATE * duration_sec)
    assert abs(out.size - expected_len) <= 2


def test_time_formatters() -> None:
    """HH:MM:SS と SRT タイムスタンプ変換のテスト"""
    assert format_hhmmss(0) == "00:00:00"
    assert format_hhmmss(3665.9) == "01:01:05"
    assert format_srt_time(65.125) == "00:01:05,125"


def test_exporters() -> None:
    """TXT / Markdown / SRT エクスポートの出力検証"""
    segs = [
        TranscriptSegment(
            index=1,
            start_sec=1.5,
            end_sec=4.2,
            text="今日の講義では線形代数について学びます。",
            wall_time="10:00:01",
        ),
        TranscriptSegment(
            index=2,
            start_sec=5.0,
            end_sec=8.0,
            text="固有値と固有ベクトルが重要です。",
            wall_time="10:00:05",
        ),
    ]

    txt = segments_to_txt(segs)
    assert "[00:00:01] 今日の講義では線形代数について学びます。" in txt

    md = segments_to_markdown(segs, title="線形代数第1回")
    assert "# 線形代数第1回" in md
    assert "**`00:00:05`** 固有値と固有ベクトルが重要です。" in md

    srt = segments_to_srt(segs)
    assert "00:00:01,500 --> 00:00:04,200" in srt

    with tempfile.TemporaryDirectory() as tmp:
        out_path = export_segments(segs, Path(tmp) / "out.srt")
        assert out_path.exists()
        assert "固有値と固有ベクトルが重要です。" in out_path.read_text(encoding="utf-8")


def test_config_roundtrip(tmp_path: Path, monkeypatch) -> None:
    """設定ファイルの保存と再読み込みが正しく動作することを検証する"""
    cfg_dir = tmp_path / ".lecture_transcriber"
    cfg_file = cfg_dir / "config.json"
    monkeypatch.setattr("src.config.CONFIG_DIR", cfg_dir)
    monkeypatch.setattr("src.config.CONFIG_FILE", cfg_file)

    cfg = AppConfig(model_size="medium", language="en", overlay_font_size=24)
    cfg.save()

    loaded = AppConfig.load()
    assert loaded.model_size == "medium"
    assert loaded.language == "en"
    assert loaded.overlay_font_size == 24


def test_join_segments_and_normalize() -> None:
    """日本語・英語混在の結合と小音量音声の正規化を検証する"""
    assert _join_segments(["今日は", "線形代数を学びます。"]) == "今日は線形代数を学びます。"
    assert _join_segments(["Hello", "world."]) == "Hello world."

    quiet = np.array([0.01, -0.02, 0.005], dtype=np.float32)
    normed = _normalize_audio(quiet)
    assert float(np.max(np.abs(normed))) > 0.1

    # プロンプトリーク防止ガードの検証
    assert _clean_text("線形代数、固有値", prompt="線形代数, 固有値, 行列式") == ""
    assert _clean_text("今日の講義は線形代数です。", prompt="線形代数, 固有値") == "今日の講義は線形代数です。"


def test_glossary_extract_from_markdown(tmp_path: Path) -> None:
    """Markdownファイルから見出し・太字・コード・専門用語を自動抽出できることを検証する"""
    md_text = """
# 機械学習とディープラーニング

## 本日のキーワード
- **誤差逆伝播法**: パラメータの勾配を計算する手法
- `Transformer` と **Attention機構** の仕組み
- 過学習を防ぐために正則化（L2正則化）を用いる。
"""
    terms = extract_keywords_from_markdown(md_text)
    assert "誤差逆伝播法" in terms
    assert "Transformer" in terms
    assert "Attention機構" in terms

    md_file = tmp_path / "lecture_notes.md"
    md_file.write_text(md_text, encoding="utf-8")
    file_terms = extract_keywords_from_file(md_file)
    assert file_terms == terms


def test_glossary_dictionary_sections_and_correction() -> None:
    """構造化単語帳の全単語読み込み・動的セクション追従・表記補正を検証する"""
    md_text = """
## 序章 先史の世界
- **ホモ＝サピエンス**：現生人類。
- **アウストラロピテクス**：アフリカで発見された猿人。

## １章 オリエントと地中海世界
### １節 古代オリエント世界
- **ハンムラビ法典**：バビロン第１王朝の法典。
- **アケメネス朝ペルシア**：古代イランの大帝国。
"""
    d = GlossaryDictionary.from_markdown_text(md_text)
    assert d.total_terms == 4
    assert len(d.sections) == 2

    # 発話内容からセクションを自動推定して関連語をプロンプトに含める
    prompt = d.build_dynamic_prompt(recent_text="ハンムラビ法典について説明します")
    assert prompt is not None
    assert "アケメネス朝ペルシア" in prompt

    # カタカナ記号ゆれ・誤認識の自動補正
    assert d.correct_text("ホモサピエンスとアウストラロピティクス") == "ホモ＝サピエンスとアウストラロピテクス"


def test_file_chunk_iterator(tmp_path: Path) -> None:
    """音声ファイルが指定チャンク秒数で正しく分割ストリーミングされることを検証する"""
    import wave

    wav_path = tmp_path / "sample.wav"
    sr = 16000
    duration = 3.0
    samples = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration))) * 16000).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())

    chunks = list(iter_media_audio_chunks(wav_path, chunk_seconds=1.2))
    assert len(chunks) == 3
    assert abs(chunks[0][0] - 0.0) < 0.05
    assert abs(chunks[1][0] - 1.2) < 0.05
    assert abs(chunks[2][0] - 2.4) < 0.05



