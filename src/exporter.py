# -*- coding: utf-8 -*-
"""文字起こし結果のデータモデルとエクスポート（TXT / Markdown / SRT）モジュール"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class TranscriptSegment:
    """確定した1つの文字起こしセグメントを表すデータクラス"""

    index: int  # 1始まりの連番
    start_sec: float  # 録音開始からの開始時刻（秒）
    end_sec: float  # 録音開始からの終了時刻（秒）
    text: str  # 文字起こしされたテキスト
    wall_time: str  # 記録された実時刻（HH:MM:SS形式）


def format_hhmmss(seconds: float) -> str:
    """秒数を HH:MM:SS 形式の文字列に変換する"""
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_srt_time(seconds: float) -> str:
    """秒数をSRT字幕フォーマット（HH:MM:SS,mmm）に変換する"""
    ms_total = max(0, int(round(seconds * 1000)))
    h = ms_total // 3_600_000
    m = (ms_total % 3_600_000) // 60_000
    s = (ms_total % 60_000) // 1000
    ms = ms_total % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_txt(segments: list[TranscriptSegment]) -> str:
    """セグメント一覧をタイムスタンプ付きプレーンテキストに変換する"""
    lines: list[str] = []
    for seg in segments:
        ts = format_hhmmss(seg.start_sec)
        lines.append(f"[{ts}] {seg.text}")
    return "\n".join(lines) + ("\n" if lines else "")


def segments_to_markdown(
    segments: list[TranscriptSegment],
    title: str = "講義文字起こしノート",
) -> str:
    """セグメント一覧をMarkdown形式の講義ノートに変換する"""
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = [
        f"# {title}",
        "",
        f"- 作成日時: {created_at}",
        f"- セグメント数: {len(segments)}",
        "",
        "## 文字起こし本文",
        "",
    ]
    for seg in segments:
        ts = format_hhmmss(seg.start_sec)
        lines.append(f"- **`{ts}`** {seg.text}")
    return "\n".join(lines) + "\n"


def segments_to_srt(segments: list[TranscriptSegment]) -> str:
    """セグメント一覧をSRT字幕ファイル形式に変換する"""
    blocks: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        start_str = format_srt_time(seg.start_sec)
        # 終了時刻が開始時刻より短い場合は最低0.5秒を確保する
        end_val = max(seg.end_sec, seg.start_sec + 0.5)
        end_str = format_srt_time(end_val)
        blocks.append(f"{idx}\n{start_str} --> {end_str}\n{seg.text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def export_segments(
    segments: list[TranscriptSegment],
    file_path: str | Path,
    fmt: str | None = None,
) -> Path:
    """指定されたファイルパスと形式（txt / md / srt）で文字起こし結果を書き出す"""
    path = Path(file_path)
    ext = (fmt or path.suffix.lstrip(".")).lower()

    if ext == "md":
        content = segments_to_markdown(segments)
    elif ext == "srt":
        content = segments_to_srt(segments)
    else:
        content = segments_to_txt(segments)

    path.write_text(content, encoding="utf-8")
    return path
