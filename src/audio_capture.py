# -*- coding: utf-8 -*-
"""クロスプラットフォーム音声キャプチャモジュール

Windowsでは PyAudioWPatch による WASAPI ループバック（PC内部音声の直接録音）と
マイク入力に対応し、macOS / Linux では sounddevice による入力デバイスおよび
BlackHole 等の仮想オーディオデバイス経由のシステム音声キャプチャに対応する。
"""

from __future__ import annotations

import math
import queue
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.signal import resample_poly

from .config import IS_WINDOWS, TARGET_SAMPLE_RATE

# Windows向け WASAPI ループバック対応ライブラリの動的インポート
try:
    if IS_WINDOWS:
        import pyaudiowpatch as pyaudio_wp  # type: ignore
    else:
        pyaudio_wp = None
except ImportError:
    pyaudio_wp = None

# クロスプラットフォーム音声入力ライブラリ
try:
    import sounddevice as sd
except ImportError:
    sd = None  # type: ignore


@dataclass
class AudioDeviceInfo:
    """音声入力デバイスの情報を保持するデータクラス"""

    id: str  # 一意識別子（設定保存用）
    name: str  # UI表示用のデバイス名
    device_index: int  # バックエンド内のデバイスインデックス
    backend: str  # "pyaudiowpatch" または "sounddevice"
    is_loopback: bool  # PC内部音声（ループバック）かどうか
    channels: int  # チャンネル数
    sample_rate: int  # デバイスのネイティブサンプルレート (Hz)
    is_default: bool = False  # 推奨デフォルトデバイスかどうか


def resample_to_16k_mono(
    audio: np.ndarray,
    src_rate: int,
    channels: int,
) -> np.ndarray:
    """任意のサンプルレート・チャンネル数の音声配列を 16kHz モノラル float32 に変換する"""
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)

    # float32 へ正規化
    data = audio.astype(np.float32, copy=False)

    # マルチチャンネルの場合はモノラルにダウンミックス
    if channels > 1:
        if data.ndim == 1:
            # インターリーブ配列を (フレーム数, チャンネル数) に整形
            frame_count = data.size // channels
            if frame_count == 0:
                return np.zeros(0, dtype=np.float32)
            data = data[: frame_count * channels].reshape(frame_count, channels)
        data = np.mean(data, axis=1)
    elif data.ndim > 1:
        data = np.mean(data, axis=1)

    # サンプルレートが 16kHz と異なる場合はポリフェーズフィルタでリサンプリング
    if src_rate != TARGET_SAMPLE_RATE and src_rate > 0:
        divisor = math.gcd(int(src_rate), TARGET_SAMPLE_RATE)
        up = TARGET_SAMPLE_RATE // divisor
        down = int(src_rate) // divisor
        data = resample_poly(data, up, down).astype(np.float32)

    return np.clip(data, -1.0, 1.0)


def _is_virtual_loopback_name(name: str) -> bool:
    """macOS / Linux 向けに仮想オーディオデバイス名からループバックかを推定する"""
    lower = name.lower()
    keywords = (
        "blackhole",
        "loopback",
        "soundflower",
        "virtual",
        "background music",
        "vb-cable",
        "cable output",
        "monitor of",
    )
    return any(kw in lower for kw in keywords)


def list_audio_devices() -> list[AudioDeviceInfo]:
    """利用可能な音声入力デバイス一覧を推奨順（システム音声優先）で返す"""
    devices: list[AudioDeviceInfo] = []

    # 1. Windows: PyAudioWPatch で WASAPI ループバックとマイクを列挙
    if IS_WINDOWS and pyaudio_wp is not None:
        pa = pyaudio_wp.PyAudio()
        try:
            default_loopback_idx: int | None = None
            try:
                default_lb = pa.get_default_wasapi_loopback()
                default_loopback_idx = int(default_lb["index"])
            except Exception:
                default_loopback_idx = None

            # ループバックデバイス（PC内部音声）を先に追加
            seen_names: set[str] = set()
            for lb in pa.get_loopback_device_info_generator():
                idx = int(lb["index"])
                raw_name = str(lb["name"]).strip()
                is_def = idx == default_loopback_idx
                prefix = "[システム音声 ★推奨]" if is_def else "[システム音声]"
                devices.append(
                    AudioDeviceInfo(
                        id=f"wasapi_lb:{idx}",
                        name=f"{prefix} {raw_name}",
                        device_index=idx,
                        backend="pyaudiowpatch",
                        is_loopback=True,
                        channels=max(1, int(lb.get("maxInputChannels", 2))),
                        sample_rate=int(lb.get("defaultSampleRate", 48000)),
                        is_default=is_def,
                    )
                )
                seen_names.add(raw_name)

            # 通常のマイク入力デバイスを追加（WASAPIホストAPI優先）
            try:
                wasapi_info = pa.get_host_api_info_by_type(pyaudio_wp.paWASAPI)
                wasapi_api_idx = int(wasapi_info["index"])
            except Exception:
                wasapi_api_idx = -1

            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if int(info.get("maxInputChannels", 0)) <= 0:
                    continue
                if info.get("isLoopbackDevice", False):
                    continue
                # WASAPIがある場合は重複を避けるためWASAPIデバイスのみ表示
                if wasapi_api_idx >= 0 and int(info.get("hostApi", -1)) != wasapi_api_idx:
                    continue
                raw_name = str(info["name"]).strip()
                if raw_name in seen_names:
                    continue
                seen_names.add(raw_name)
                devices.append(
                    AudioDeviceInfo(
                        id=f"wasapi_in:{i}",
                        name=f"[マイク/入力] {raw_name}",
                        device_index=i,
                        backend="pyaudiowpatch",
                        is_loopback=False,
                        channels=max(1, int(info.get("maxInputChannels", 1))),
                        sample_rate=int(info.get("defaultSampleRate", 48000)),
                        is_default=False,
                    )
                )
        finally:
            pa.terminate()

        if devices:
            # デフォルト推奨デバイスを先頭に並び替え
            devices.sort(key=lambda d: (not d.is_default, not d.is_loopback))
            return devices

    # 2. macOS / Linux / フォールバック: sounddevice を使用
    if sd is not None:
        all_devs = sd.query_devices()
        try:
            default_in_idx = sd.default.device[0]
        except Exception:
            default_in_idx = -1

        virtual_devices: list[AudioDeviceInfo] = []
        mic_devices: list[AudioDeviceInfo] = []

        for idx, info in enumerate(all_devs):
            max_in = int(info.get("max_input_channels", 0))
            if max_in <= 0:
                continue
            raw_name = str(info["name"]).strip()
            is_virtual = _is_virtual_loopback_name(raw_name)
            is_def = (idx == default_in_idx) and not is_virtual
            sr = int(info.get("default_samplerate", 48000))

            dev_obj = AudioDeviceInfo(
                id=f"sd:{idx}",
                name=(
                    f"[仮想音声/システム音声] {raw_name}"
                    if is_virtual
                    else f"[マイク/入力] {raw_name}"
                ),
                device_index=idx,
                backend="sounddevice",
                is_loopback=is_virtual,
                channels=min(max_in, 2),
                sample_rate=sr,
                is_default=is_virtual or is_def,
            )
            if is_virtual:
                virtual_devices.append(dev_obj)
            else:
                mic_devices.append(dev_obj)

        # 仮想ループバックデバイス（BlackHole等）があれば先頭に配置
        devices = virtual_devices + mic_devices

    return devices


class AudioCaptureStream:
    """選択された音声デバイスからリアルタイムで16kHzモノラル音声を取得するストリーム"""

    def __init__(
        self,
        device: AudioDeviceInfo,
        audio_queue: queue.Queue[np.ndarray],
        level_callback: Callable[[float], None] | None = None,
    ) -> None:
        self.device = device
        self.audio_queue = audio_queue
        self.level_callback = level_callback

        self._running = False
        self._paused = False
        self._pa = None
        self._pa_stream = None
        self._sd_stream = None
        self._lock = threading.Lock()

    def set_paused(self, paused: bool) -> None:
        """音声取り込みの一時停止・再開を切り替える"""
        self._paused = paused

    def start(self) -> None:
        """音声キャプチャを開始する"""
        with self._lock:
            if self._running:
                return
            self._running = True

            if self.device.backend == "pyaudiowpatch" and pyaudio_wp is not None:
                self._start_pyaudio_wp()
            elif sd is not None:
                self._start_sounddevice()
            else:
                self._running = False
                raise RuntimeError("利用可能な音声バックエンドが見つかりません。")

    def stop(self) -> None:
        """音声キャプチャを安全に停止・解放する"""
        with self._lock:
            self._running = False
            if self._pa_stream is not None:
                try:
                    self._pa_stream.stop_stream()
                    self._pa_stream.close()
                except Exception:
                    pass
                self._pa_stream = None

            if self._pa is not None:
                try:
                    self._pa.terminate()
                except Exception:
                    pass
                self._pa = None

            if self._sd_stream is not None:
                try:
                    self._sd_stream.stop()
                    self._sd_stream.close()
                except Exception:
                    pass
                self._sd_stream = None

    def _emit_audio(self, mono_16k: np.ndarray) -> None:
        """変換済み16kHz音声をキューへ送り、音量レベルを通知する"""
        if not self._running or self._paused or mono_16k.size == 0:
            return
        # RMS音量を計算してUIメーターへ通知
        if self.level_callback is not None:
            rms = float(np.sqrt(np.mean(np.square(mono_16k))))
            self.level_callback(rms)

        try:
            self.audio_queue.put_nowait(mono_16k)
        except queue.Full:
            # キューが満杯の場合は古いチャンクを捨てて最新音声を優先
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(mono_16k)
            except Exception:
                pass

    def _start_pyaudio_wp(self) -> None:
        """Windows PyAudioWPatch (WASAPI) ストリームを開始する"""
        self._pa = pyaudio_wp.PyAudio()
        channels = max(1, self.device.channels)
        rate = max(8000, self.device.sample_rate)
        # 約100msごとのフレーム数
        frames_per_buffer = max(512, int(rate * 0.1))

        def callback(in_data, frame_count, time_info, status):
            if not self._running:
                return (None, pyaudio_wp.paComplete)
            raw = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            mono_16k = resample_to_16k_mono(raw, rate, channels)
            self._emit_audio(mono_16k)
            return (None, pyaudio_wp.paContinue)

        self._pa_stream = self._pa.open(
            format=pyaudio_wp.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=self.device.device_index,
            frames_per_buffer=frames_per_buffer,
            stream_callback=callback,
        )
        self._pa_stream.start_stream()

    def _start_sounddevice(self) -> None:
        """macOS / Linux / フォールバック用 sounddevice ストリームを開始する"""
        channels = max(1, self.device.channels)
        rate = max(8000, self.device.sample_rate)
        blocksize = max(512, int(rate * 0.1))

        def callback(indata, frames, time_info, status):
            if not self._running:
                return
            mono_16k = resample_to_16k_mono(indata.copy(), rate, channels)
            self._emit_audio(mono_16k)

        self._sd_stream = sd.InputStream(
            device=self.device.device_index,
            channels=channels,
            samplerate=rate,
            dtype="float32",
            blocksize=blocksize,
            callback=callback,
        )
        self._sd_stream.start()
