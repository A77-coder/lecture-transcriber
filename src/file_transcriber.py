# -*- coding: utf-8 -*-
"""長時間対応の動画・音声ファイル直接文字起こしモジュール

PyAV (FFmpeg内蔵) を用いて動画・音声ファイルから16kHzモノラル音声を
チャンク単位（デフォルト10分ごと）でストリーミング抽出し、
数時間に及ぶ講義動画でも低メモリ・安定動作で高精度に文字起こしを行う。
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

import av
import numpy as np

from .config import AppConfig, TARGET_SAMPLE_RATE
from .exporter import TranscriptSegment, format_hhmmss
from .glossary import GlossaryDictionary
from .transcriber import _clean_text, _normalize_audio, detect_compute_device

# 動画・音声として受け入れる拡張子一覧
SUPPORTED_MEDIA_EXTENSIONS = (
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".m4v",
    ".wmv",
    ".flv",
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
    ".wma",
    ".opus",
)


def get_media_duration_sec(file_path: str | Path) -> float:
    """メディアファイルの総再生時間（秒）を取得する（不明時は0.0）"""
    try:
        with av.open(str(file_path)) as container:
            if container.duration is not None and container.duration > 0:
                return float(container.duration) / float(av.time_base)
            audio_streams = [s for s in container.streams if s.type == "audio"]
            if audio_streams:
                stream = audio_streams[0]
                if stream.duration is not None and stream.time_base is not None:
                    return float(stream.duration * stream.time_base)
    except Exception:
        pass
    return 0.0


def iter_media_audio_chunks(
    file_path: str | Path,
    chunk_seconds: float = 600.0,
) -> Iterator[tuple[float, np.ndarray]]:
    """動画/音声ファイルから16kHzモノラル音声をchunk_seconds秒ごとにストリーミング出力する

    Returns:
        (チャンク開始オフセット秒, float32モノラル16kHz音声配列) のタプルを順次yieldする
    """
    chunk_samples = int(TARGET_SAMPLE_RATE * chunk_seconds)
    resampler = av.audio.resampler.AudioResampler(
        format="fltp",
        layout="mono",
        rate=TARGET_SAMPLE_RATE,
    )

    buffer_parts: list[np.ndarray] = []
    buffered_samples = 0
    total_emitted_samples = 0

    with av.open(str(file_path)) as container:
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
            raise ValueError("このファイルには音声トラックが含まれていません。")
        stream = audio_streams[0]
        stream.thread_type = "AUTO"

        for frame in container.decode(stream):
            resampled_frames = resampler.resample(frame)
            if not resampled_frames:
                continue
            for rf in resampled_frames:
                arr = rf.to_ndarray()
                if arr.ndim > 1:
                    arr = arr[0]
                arr = arr.astype(np.float32, copy=False)
                if arr.size == 0:
                    continue
                buffer_parts.append(arr)
                buffered_samples += arr.size

                while buffered_samples >= chunk_samples:
                    combined = np.concatenate(buffer_parts)
                    chunk_audio = combined[:chunk_samples]
                    remainder = combined[chunk_samples:]
                    buffer_parts = [remainder] if remainder.size > 0 else []
                    buffered_samples = remainder.size

                    offset_sec = total_emitted_samples / TARGET_SAMPLE_RATE
                    total_emitted_samples += chunk_audio.size
                    yield (offset_sec, chunk_audio)

        # リサンプラー内部に残ったフラッシュフレームを取り出す
        for rf in resampler.resample(None):
            arr = rf.to_ndarray()
            if arr.ndim > 1:
                arr = arr[0]
            arr = arr.astype(np.float32, copy=False)
            if arr.size > 0:
                buffer_parts.append(arr)
                buffered_samples += arr.size

    if buffer_parts and buffered_samples > 0:
        tail = np.concatenate(buffer_parts)
        offset_sec = total_emitted_samples / TARGET_SAMPLE_RATE
        yield (offset_sec, tail)


class FileBatchTranscriber:
    """動画・音声ファイルをバックグラウンドでチャンク分割文字起こしするワーカークラス"""

    def __init__(
        self,
        file_path: str | Path,
        config: AppConfig,
        glossary: GlossaryDictionary | None = None,
        chunk_seconds: float = 600.0,
        on_segment: Callable[[TranscriptSegment], None] | None = None,
        on_progress: Callable[[float, float], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        on_finished: Callable[[bool], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.config = config
        self.glossary = glossary
        self.chunk_seconds = chunk_seconds
        self.on_segment = on_segment
        self.on_progress = on_progress
        self.on_status = on_status
        self.on_finished = on_finished
        self.on_error = on_error

        self._model = None
        self._device_label = "AUTO"
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._segment_counter = 0
        self._last_context_text = ""

    def start(self, initial_segment_index: int = 0) -> None:
        """ファイル文字起こしワーカーを開始する"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._pause_event.clear()
        self._segment_counter = initial_segment_index
        self._thread = threading.Thread(
            target=self._run,
            name="FileWhisperWorker",
            daemon=True,
        )
        self._thread.start()

    def set_paused(self, paused: bool) -> None:
        """ファイル文字起こしの一時停止・再開を切り替える"""
        if paused:
            self._pause_event.set()
            if self.on_status:
                self.on_status("ファイル文字起こしを一時停止中")
        else:
            self._pause_event.clear()
            if self.on_status:
                self.on_status(
                    f"ファイル文字起こし中 ({self._device_label}): {self.file_path.name}"
                )

    def stop(self) -> None:
        """ファイル文字起こしを中断する"""
        self._stop_event.set()
        self._pause_event.clear()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _load_model(self) -> None:
        """設定に応じたWhisperモデルをロードする（CUDA失敗時はCPUフォールバック）"""
        device, compute_type = detect_compute_device(self.config.device_preference)
        if self.on_status:
            self.on_status(
                f"モデル '{self.config.model_size}' を読み込み中 ({device.upper()} / {compute_type})... ※初回はダウンロードに数分かかります"
            )
        from faster_whisper import WhisperModel

        try:
            self._model = WhisperModel(
                self.config.model_size,
                device=device,
                compute_type=compute_type,
            )
            self._device_label = device.upper()
        except Exception as exc:
            if device == "cuda":
                if self.on_status:
                    self.on_status(f"GPU読み込みに失敗したためCPUへ切り替えます: {exc}")
                self._model = WhisperModel(
                    self.config.model_size,
                    device="cpu",
                    compute_type="int8",
                )
                self._device_label = "CPU"
            else:
                raise

    def _build_prompt(self) -> str | None:
        """現在の文脈と全単語辞書から最適なinitial_promptを生成する"""
        if self.glossary is not None and self.glossary.total_terms > 0:
            return self.glossary.build_dynamic_prompt(
                selected_section=self.config.glossary_section,
                recent_text=self._last_context_text,
                manual_hint=self.config.custom_prompt,
            )
        parts: list[str] = []
        if self.config.custom_prompt.strip():
            parts.append(self.config.custom_prompt.strip()[:180])
        if self._last_context_text:
            parts.append(self._last_context_text[-60:])
        prompt = " ".join(parts).strip()
        return prompt if prompt else None

    def _run(self) -> None:
        """チャンク単位で音声をデコード＆文字起こしするメインループ"""
        try:
            self._load_model()
            total_sec = get_media_duration_sec(self.file_path)
            if self.on_status:
                dur_str = format_hhmmss(total_sec) if total_sec > 0 else "長さ計測中"
                self.on_status(
                    f"ファイル文字起こし中 ({self._device_label}): {self.file_path.name} ({dur_str})"
                )

            lang = None if self.config.language == "auto" else self.config.language
            vad_params = {"min_silence_duration_ms": 450, "speech_pad_ms": 300}

            for chunk_offset, chunk_audio in iter_media_audio_chunks(
                self.file_path, chunk_seconds=self.chunk_seconds
            ):
                if self._stop_event.is_set():
                    break

                # 一時停止中は再開またはキャンセルまで待機
                while self._pause_event.is_set() and not self._stop_event.is_set():
                    time.sleep(0.15)
                if self._stop_event.is_set():
                    break

                norm_audio = _normalize_audio(chunk_audio)
                prompt = self._build_prompt()

                segments_iter, _ = self._model.transcribe(
                    norm_audio,
                    language=lang,
                    beam_size=5,
                    initial_prompt=prompt,
                    vad_filter=True,
                    vad_parameters=vad_params,
                    condition_on_previous_text=True,
                    repetition_penalty=1.1,
                    no_speech_threshold=0.6,
                )

                for seg in segments_iter:
                    if self._stop_event.is_set():
                        break
                    while self._pause_event.is_set() and not self._stop_event.is_set():
                        time.sleep(0.15)
                    if self._stop_event.is_set():
                        break

                    abs_start = chunk_offset + float(seg.start)
                    abs_end = chunk_offset + float(seg.end)
                    text = _clean_text(seg.text, prompt=prompt)
                    if text and self.glossary is not None and self.glossary.total_terms > 0:
                        text = self.glossary.correct_text(text)

                    if text:
                        self._last_context_text = text
                        self._segment_counter += 1
                        item = TranscriptSegment(
                            index=self._segment_counter,
                            start_sec=abs_start,
                            end_sec=abs_end,
                            text=text,
                            wall_time=datetime.now().strftime("%H:%M:%S"),
                        )
                        if self.on_segment:
                            self.on_segment(item)

                    if self.on_progress:
                        self.on_progress(abs_end, max(total_sec, abs_end))

            completed = not self._stop_event.is_set()
            if completed and self.on_progress and total_sec > 0:
                self.on_progress(total_sec, total_sec)
            if self.on_status:
                self.on_status(
                    f"ファイル文字起こし完了: {self.file_path.name}"
                    if completed
                    else "ファイル文字起こしを中断しました"
                )
            if self.on_finished:
                self.on_finished(completed)
        except Exception as exc:
            if self.on_error:
                self.on_error(f"ファイル文字起こし中にエラーが発生しました:\n{exc}")
