# -*- coding: utf-8 -*-
"""リアルタイム音声文字起こしエンジンモジュール (faster-whisperベース)"""

from __future__ import annotations

import os
import queue
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from .config import AppConfig, IS_WINDOWS, TARGET_SAMPLE_RATE
from .exporter import TranscriptSegment
from .glossary import GlossaryDictionary


_DLL_DIR_HANDLES: list[object] = []


def _register_nvidia_dll_paths() -> None:
    """Windows環境でpipインストールされたNVIDIA CUDA DLLを探索パスとPATHに追加する"""
    if not IS_WINDOWS:
        return
    try:
        added_dirs: list[str] = []
        # 現在のPython環境のsite-packages配下にあるnvidia/*/binを登録
        for base in sys.path:
            nvidia_dir = Path(base) / "nvidia"
            if nvidia_dir.is_dir():
                for bin_dir in nvidia_dir.glob("*/bin"):
                    if bin_dir.is_dir():
                        dir_str = str(bin_dir)
                        if dir_str not in added_dirs:
                            added_dirs.append(dir_str)
                            # ハンドルを保持してGCによる登録解除を防ぐ
                            _DLL_DIR_HANDLES.append(os.add_dll_directory(dir_str))
        if added_dirs:
            current_path = os.environ.get("PATH", "")
            missing = [d for d in added_dirs if d not in current_path]
            if missing:
                os.environ["PATH"] = os.pathsep.join(missing) + os.pathsep + current_path
    except Exception:
        pass


def detect_compute_device(preference: str = "auto") -> tuple[str, str]:
    """使用する推論デバイス(cuda/cpu)と計算精度(int8_float16/int8)を自動判定する"""
    if preference == "cpu":
        return ("cpu", "int8")

    _register_nvidia_dll_paths()
    try:
        import ctranslate2

        cuda_count = ctranslate2.get_cuda_device_count()
        if cuda_count > 0 and preference in ("auto", "cuda"):
            # VRAM使用量を抑えつつ高速・高精度に動作する int8_float16 を優先
            return ("cuda", "int8_float16")
    except Exception:
        pass

    return ("cpu", "int8")


# 無音時にWhisperが出力しがちな代表的な定型ハルシネーション句
_HALLUCINATION_PHRASES = {
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございました。",
    "チャンネル登録よろしくお願いします",
    "チャンネル登録よろしくお願いします。",
    "お疲れ様でした",
    "お疲れ様でした。",
    "Thank you for watching.",
    "Thanks for watching!",
}


def _clean_text(text: str, prompt: str | None = None) -> str:
    """文字起こし結果の前後空白を除去し、無音ハルシネーションやプロンプトリークを除外する"""
    cleaned = text.strip()
    if not cleaned:
        return ""
    if cleaned in _HALLUCINATION_PHRASES:
        return ""
    # Whisperがinitial_promptの内容をそのまま出力してしまうプロンプトリークを除外
    if prompt:
        norm_out = re.sub(r"[\s、。,.]+", "", cleaned)
        norm_prompt = re.sub(r"[\s、。,.]+", "", prompt)
        if len(norm_out) >= 4 and norm_out in norm_prompt:
            return ""
    return cleaned


_JP_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


def _join_segments(texts: list[str], prompt: str | None = None) -> str:
    """複数セグメントを言語に応じて自然に結合する（日本語間はスペースを入れない）"""
    if not texts:
        return ""
    result = texts[0]
    for nxt in texts[1:]:
        if _JP_CHAR_RE.search(result[-1:]) or _JP_CHAR_RE.search(nxt[:1]):
            result += nxt
        else:
            result += " " + nxt
    return _clean_text(result, prompt=prompt)


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    """小音量音声でのVAD誤判定・認識低下を防ぐため適度にゲイン正規化する"""
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if 0.001 < peak < 0.25:
        gain = min(8.0, 0.55 / peak)
        return np.clip(audio * gain, -1.0, 1.0)
    return audio


class RealtimeTranscriber:
    """音声キューからチャンクを受け取り、リアルタイムで文字起こしを行うワーカークラス"""

    def __init__(
        self,
        config: AppConfig,
        audio_queue: queue.Queue[np.ndarray],
        glossary: GlossaryDictionary | None = None,
        on_partial: Callable[[str], None] | None = None,
        on_final: Callable[[TranscriptSegment], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.audio_queue = audio_queue
        self.glossary = glossary
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_status = on_status
        self.on_error = on_error

        self._model = None
        self._loaded_model_key: tuple[str, str, str] | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._paused_at: float | None = None
        self._worker_thread: threading.Thread | None = None
        self._session_start_time: float = 0.0
        self._segment_counter: int = 0
        self._last_context_text: str = ""  # 直前の確定文（文脈引き継ぎ用）

    def set_paused(self, paused: bool) -> None:
        """文字起こしの一時停止・再開を切り替える（タイムスタンプ補正付き）"""
        if paused and not self._pause_event.is_set():
            self._paused_at = time.monotonic()
            self._pause_event.set()
            if self.on_status:
                self.on_status("一時停止中")
        elif not paused and self._pause_event.is_set():
            if self._paused_at is not None:
                # 一時停止していた時間だけ開始時刻をずらしてタイムスタンプを連続させる
                self._session_start_time += time.monotonic() - self._paused_at
                self._paused_at = None
            self._pause_event.clear()
            if self.on_status:
                dev_label = (
                    self._loaded_model_key[1].upper()
                    if self._loaded_model_key
                    else "AUTO"
                )
                self.on_status(f"リアルタイム文字起こし中 ({dev_label})")

    def ensure_model_loaded(self) -> tuple[str, str]:
        """設定に応じたWhisperモデルをロードする（未ロードまたは設定変更時のみ）"""
        device, compute_type = detect_compute_device(self.config.device_preference)
        key = (self.config.model_size, device, compute_type)

        if self._model is not None and self._loaded_model_key == key:
            return (device, compute_type)

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
            self._loaded_model_key = key
        except Exception as exc:
            # CUDA初期化に失敗した場合はCPU(int8)へ自動フォールバックする
            if device == "cuda":
                if self.on_status:
                    self.on_status(
                        f"GPU読み込みに失敗したためCPUモードへ切り替えます: {exc}"
                    )
                device, compute_type = ("cpu", "int8")
                self._model = WhisperModel(
                    self.config.model_size,
                    device=device,
                    compute_type=compute_type,
                )
                self._loaded_model_key = (self.config.model_size, device, compute_type)
            else:
                raise

        if self.on_status:
            self.on_status(
                f"文字起こし待機中 (モデル: {self.config.model_size} / {device.upper()})"
            )
        return (device, compute_type)

    def start(self, initial_segment_index: int = 0) -> None:
        """文字起こしワーカースレッドを開始する"""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._session_start_time = time.monotonic()
        self._segment_counter = initial_segment_index
        self._worker_thread = threading.Thread(
            target=self._run_loop,
            name="WhisperWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def stop(self) -> None:
        """文字起こしワーカースレッドを停止する"""
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
            self._worker_thread = None

    def _fallback_to_cpu(self, reason: Exception) -> None:
        """GPU推論に失敗した場合にCPU(int8)モデルへ切り替える"""
        if self.on_status:
            self.on_status(f"GPU推論エラーのためCPUモードへ切り替えます: {reason}")
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.config.model_size,
            device="cpu",
            compute_type="int8",
        )
        self._loaded_model_key = (self.config.model_size, "cpu", "int8")
        if self.on_status:
            self.on_status("リアルタイム文字起こし中 (CPU)")

    def _build_initial_prompt(self) -> str | None:
        """全単語辞書・手動ヒント・直前文脈から最適なinitial_promptを動的生成する

        ※指示文（「〜してください」等）はWhisperがそのまま出力してしまうため含めない。
        """
        if self.glossary is not None and self.glossary.total_terms > 0:
            return self.glossary.build_dynamic_prompt(
                selected_section=self.config.glossary_section,
                recent_text=self._last_context_text,
                manual_hint=self.config.custom_prompt,
            )

        parts: list[str] = []
        custom = self.config.custom_prompt.strip()
        if custom:
            parts.append(custom[:180])
        if self._last_context_text:
            parts.append(self._last_context_text[-60:])
        prompt = " ".join(parts).strip()
        return prompt if prompt else None

    def _transcribe_array(self, audio: np.ndarray, is_final: bool) -> str:
        """16kHzモノラル音声配列をWhisperで推論しテキストを返す"""
        if self._model is None or audio.size < int(TARGET_SAMPLE_RATE * 0.35):
            return ""

        # 音量を正規化し、語尾の欠落を防ぐため末尾に0.3秒の無音パディングを追加
        norm_audio = _normalize_audio(audio)
        pad = np.zeros(int(TARGET_SAMPLE_RATE * 0.3), dtype=np.float32)
        prepared_audio = np.concatenate([norm_audio, pad])

        lang = None if self.config.language == "auto" else self.config.language
        beam_size = 5 if is_final else 1
        vad_params = {"min_silence_duration_ms": 400, "speech_pad_ms": 300}
        prompt = self._build_initial_prompt()

        def _run_infer() -> str:
            segments, _ = self._model.transcribe(
                prepared_audio,
                language=lang,
                beam_size=beam_size,
                initial_prompt=prompt,
                vad_filter=is_final,
                vad_parameters=vad_params if is_final else None,
                condition_on_previous_text=False,
                repetition_penalty=1.1,
                no_speech_threshold=0.6,
            )
            texts = [seg.text.strip() for seg in segments if seg.text.strip()]
            joined = _join_segments(texts, prompt=prompt)
            if joined and self.glossary is not None and self.glossary.total_terms > 0:
                joined = self.glossary.correct_text(joined)
            return joined

        try:
            return _run_infer()
        except Exception as exc:
            # CUDA実行時エラー（DLL不足等）の場合はCPUへフォールバックして再試行する
            if self._loaded_model_key and self._loaded_model_key[1] == "cuda":
                self._fallback_to_cpu(exc)
                return _run_infer()
            raise

    def _run_loop(self) -> None:
        """音声バッファを監視し、暫定・確定の文字起こしを順次実行するメインループ"""
        try:
            device, _ = self.ensure_model_loaded()
            if self.on_status:
                self.on_status(f"リアルタイム文字起こし中 ({device.upper()})")
        except Exception as exc:
            if self.on_error:
                self.on_error(f"モデルの読み込みに失敗しました: {exc}")
            return

        # 発話開始直前の語頭欠落を防ぐため約0.4秒分のプリロールを保持する
        preroll_chunks: deque[np.ndarray] = deque(maxlen=4)
        buffer_chunks: list[np.ndarray] = []
        buffer_samples = 0
        has_voice = False
        speech_start_elapsed = 0.0
        last_voice_time = time.monotonic()
        last_partial_time = time.monotonic()

        while not self._stop_event.is_set():
            # 一時停止中は未確定のバッファがあれば確定させ、再開まで待機する
            if self._pause_event.is_set():
                if has_voice and buffer_chunks:
                    audio_data = np.concatenate(buffer_chunks)
                    buffer_chunks.clear()
                    buffer_samples = 0
                    has_voice = False
                    text = self._transcribe_array(audio_data, is_final=True)
                    if self.on_partial:
                        self.on_partial("")
                    if text:
                        self._last_context_text = text
                    if text and self.on_final:
                        self._segment_counter += 1
                        end_elapsed = max(
                            speech_start_elapsed + 0.5,
                            (self._paused_at or time.monotonic()) - self._session_start_time,
                        )
                        seg = TranscriptSegment(
                            index=self._segment_counter,
                            start_sec=speech_start_elapsed,
                            end_sec=end_elapsed,
                            text=text,
                            wall_time=datetime.now().strftime("%H:%M:%S"),
                        )
                        self.on_final(seg)
                preroll_chunks.clear()
                time.sleep(0.1)
                continue

            got_chunk = False
            try:
                chunk = self.audio_queue.get(timeout=0.15)
                got_chunk = True
            except queue.Empty:
                chunk = None

            now = time.monotonic()

            if got_chunk and chunk is not None and chunk.size > 0:
                rms = float(np.sqrt(np.mean(np.square(chunk))))
                is_speech = rms >= self.config.silence_threshold

                if is_speech:
                    if not has_voice:
                        has_voice = True
                        speech_start_elapsed = max(0.0, now - self._session_start_time)
                        # 発話開始時に直前のプリロール音声を先頭へ結合する
                        for pre in preroll_chunks:
                            buffer_chunks.append(pre)
                            buffer_samples += pre.size
                        preroll_chunks.clear()
                    last_voice_time = now
                    buffer_chunks.append(chunk)
                    buffer_samples += chunk.size
                elif has_voice:
                    # 発話中の短い息継ぎ無音も文脈としてバッファに含める
                    buffer_chunks.append(chunk)
                    buffer_samples += chunk.size
                else:
                    preroll_chunks.append(chunk)

            if not has_voice:
                continue

            buffer_sec = buffer_samples / TARGET_SAMPLE_RATE
            silence_elapsed = now - last_voice_time

            # 確定条件: 無音が一定時間続いた、または最大セグメント長に達した
            should_finalize = (
                silence_elapsed >= self.config.silence_duration_sec
                or buffer_sec >= self.config.max_segment_sec
            )

            if should_finalize:
                audio_data = np.concatenate(buffer_chunks)
                buffer_chunks.clear()
                buffer_samples = 0
                has_voice = False

                text = self._transcribe_array(audio_data, is_final=True)
                if self.on_partial:
                    self.on_partial("")
                if text:
                    self._last_context_text = text
                if text and self.on_final:
                    self._segment_counter += 1
                    end_elapsed = max(
                        speech_start_elapsed + 0.5,
                        now - self._session_start_time,
                    )
                    seg = TranscriptSegment(
                        index=self._segment_counter,
                        start_sec=speech_start_elapsed,
                        end_sec=end_elapsed,
                        text=text,
                        wall_time=datetime.now().strftime("%H:%M:%S"),
                    )
                    self.on_final(seg)
                last_partial_time = now
                continue

            # 暫定テキスト更新条件: 一定間隔ごとに現在の発話バッファを高速推論して仮表示
            if (
                buffer_sec >= 1.0
                and (now - last_partial_time) >= self.config.partial_interval_sec
            ):
                audio_data = np.concatenate(buffer_chunks)
                partial_text = self._transcribe_array(audio_data, is_final=False)
                if partial_text and self.on_partial:
                    self.on_partial(partial_text)
                last_partial_time = now

        # 停止時に未確定の発話が残っていれば最後に確定させる
        if has_voice and buffer_chunks:
            audio_data = np.concatenate(buffer_chunks)
            text = self._transcribe_array(audio_data, is_final=True)
            if self.on_partial:
                self.on_partial("")
            if text and self.on_final:
                self._segment_counter += 1
                end_elapsed = max(
                    speech_start_elapsed + 0.5,
                    time.monotonic() - self._session_start_time,
                )
                seg = TranscriptSegment(
                    index=self._segment_counter,
                    start_sec=speech_start_elapsed,
                    end_sec=end_elapsed,
                    text=text,
                    wall_time=datetime.now().strftime("%H:%M:%S"),
                )
                self.on_final(seg)
