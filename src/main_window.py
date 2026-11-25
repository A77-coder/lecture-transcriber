# -*- coding: utf-8 -*-
"""メインアプリケーションウィンドウ (PySide6 GUI)"""

from __future__ import annotations

from pathlib import Path
import queue
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import __app_name__, __version__
from .audio_capture import AudioCaptureStream, AudioDeviceInfo, list_audio_devices
from .config import (
    AVAILABLE_LANGUAGES,
    AVAILABLE_MODELS,
    AppConfig,
    IS_MAC,
)
from .exporter import TranscriptSegment, export_segments, format_hhmmss
from .file_transcriber import FileBatchTranscriber, SUPPORTED_MEDIA_EXTENSIONS
from .glossary import GlossaryDictionary
from .subtitle_overlay import SubtitleOverlayWindow
from .transcriber import RealtimeTranscriber


class _UiBridge(QObject):
    """ワーカースレッドからのイベントをQtメインスレッドへ安全に橋渡しするシグナル集"""

    partial_received = Signal(str)
    final_received = Signal(object)
    status_received = Signal(str)
    error_received = Signal(str)
    level_received = Signal(float)
    file_progress_received = Signal(float, float)
    file_finished_received = Signal(bool)


class MainWindow(QMainWindow):
    """講義動画リアルタイム文字起こしアプリのメインウィンドウ"""

    def __init__(self) -> None:
        super().__init__()
        self.config = AppConfig.load()
        self.devices: list[AudioDeviceInfo] = []
        self.segments: list[TranscriptSegment] = []
        self.glossary = GlossaryDictionary()

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._capture_stream: AudioCaptureStream | None = None
        self._transcriber: RealtimeTranscriber | None = None
        self._file_transcriber: FileBatchTranscriber | None = None
        self._is_recording = False
        self._is_file_running = False
        self._is_paused = False
        self._record_start_time: float = 0.0
        self._paused_at: float | None = None
        self._current_level: float = 0.0

        self._bridge = _UiBridge()
        self._bridge.partial_received.connect(self._on_partial_text)
        self._bridge.final_received.connect(self._on_final_segment)
        self._bridge.status_received.connect(self._on_status_message)
        self._bridge.error_received.connect(self._on_error_message)
        self._bridge.level_received.connect(self._on_audio_level)
        self._bridge.file_progress_received.connect(self._on_file_progress)
        self._bridge.file_finished_received.connect(self._on_file_finished)

        self._overlay: SubtitleOverlayWindow | None = None

        self.setWindowTitle(f"{__app_name__} v{__version__}")
        self.resize(1040, 760)
        self.setMinimumSize(820, 600)
        # .md/.txt (単語帳) および 動画・音声ファイルのドラッグ＆ドロップに対応
        self.setAcceptDrops(True)

        self._setup_ui()
        self._apply_dark_theme()
        self._refresh_devices()
        self._reload_glossary_dictionary(silent=True)

        # 経過時間と音量メーターの平滑表示用タイマー
        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(100)
        self._ui_timer.timeout.connect(self._on_ui_tick)
        self._ui_timer.start()

    def _setup_ui(self) -> None:
        """メイン画面のウィジェットとレイアウトを構築する"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 14, 16, 10)
        main_layout.setSpacing(12)

        # --- ヘッダー＆操作パネル ---
        control_card = QFrame()
        control_card.setObjectName("Card")
        control_layout = QVBoxLayout(control_card)
        control_layout.setContentsMargins(14, 12, 14, 12)
        control_layout.setSpacing(10)

        # 1行目: 音声入力デバイス選択 + 更新ボタン + 音量メーター
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row1.addWidget(QLabel("音声ソース:"))

        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(340)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        row1.addWidget(self.device_combo, stretch=1)

        self.btn_refresh_dev = QPushButton("↻ 更新")
        self.btn_refresh_dev.setToolTip("音声デバイス一覧を再取得する")
        self.btn_refresh_dev.clicked.connect(self._refresh_devices)
        row1.addWidget(self.btn_refresh_dev)

        row1.addWidget(QLabel("音量:"))
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedWidth(120)
        self.level_bar.setFixedHeight(12)
        row1.addWidget(self.level_bar)

        control_layout.addLayout(row1)

        # 2行目: モデル・言語選択・開始/停止・字幕モード
        row2 = QHBoxLayout()
        row2.setSpacing(10)

        row2.addWidget(QLabel("AIモデル:"))
        self.model_combo = QComboBox()
        for label, value in AVAILABLE_MODELS:
            self.model_combo.addItem(label, value)
        self._select_combo_by_data(self.model_combo, self.config.model_size)
        self.model_combo.currentIndexChanged.connect(self._on_settings_changed)
        row2.addWidget(self.model_combo)

        row2.addWidget(QLabel("言語:"))
        self.lang_combo = QComboBox()
        for label, value in AVAILABLE_LANGUAGES:
            self.lang_combo.addItem(label, value)
        self._select_combo_by_data(self.lang_combo, self.config.language)
        self.lang_combo.currentIndexChanged.connect(self._on_settings_changed)
        row2.addWidget(self.lang_combo)

        row2.addStretch()

        self.btn_overlay = QPushButton("💬 字幕")
        self.btn_overlay.setCheckable(True)
        self.btn_overlay.setToolTip("動画の上に重ねて表示できる最前面の字幕ウィンドウを開きます")
        self.btn_overlay.clicked.connect(self._toggle_overlay)
        row2.addWidget(self.btn_overlay)

        self.btn_file_transcribe = QPushButton("🎬 動画/音声読込")
        self.btn_file_transcribe.setToolTip(
            "動画・音声ファイル（長時間対応）を直接読み込み、音を出さずに高精度で全文文字起こしします"
        )
        self.btn_file_transcribe.clicked.connect(self._open_media_file_dialog)
        row2.addWidget(self.btn_file_transcribe)

        self.btn_pause = QPushButton("⏸ 一時停止")
        self.btn_pause.setEnabled(False)
        self.btn_pause.setMinimumWidth(105)
        self.btn_pause.setToolTip("リアルタイム録音・ファイル文字起こしを一時停止／再開します")
        self.btn_pause.clicked.connect(self._toggle_pause)
        row2.addWidget(self.btn_pause)

        self.btn_start_stop = QPushButton("▶ 文字起こし開始")
        self.btn_start_stop.setObjectName("PrimaryButton")
        self.btn_start_stop.setMinimumWidth(150)
        self.btn_start_stop.clicked.connect(self._toggle_recording)
        row2.addWidget(self.btn_start_stop)

        control_layout.addLayout(row2)

        # 3行目: 全単語辞書（MD/TXT単語帳の読み込み・セクション絞り込み）
        row3 = QHBoxLayout()
        row3.setSpacing(8)
        row3.addWidget(QLabel("単語帳辞書:"))

        self.glossary_status_label = QLabel("未読込（MD/TXTファイルを読み込むと全単語を自動認識・補正します）")
        self.glossary_status_label.setStyleSheet("color: #94A3B8;")
        row3.addWidget(self.glossary_status_label, stretch=1)

        row3.addWidget(QLabel("範囲:"))
        self.section_combo = QComboBox()
        self.section_combo.setMinimumWidth(220)
        self.section_combo.addItem("すべて (会話から自動追従)", "")
        self.section_combo.currentIndexChanged.connect(self._on_settings_changed)
        row3.addWidget(self.section_combo)

        self.btn_load_md = QPushButton("📄 単語帳MD読込")
        self.btn_load_md.setToolTip(
            "単語帳や講義資料のMarkdown (.md) / テキスト (.txt) から全単語を辞書登録します"
        )
        self.btn_load_md.clicked.connect(self._load_glossary_dialog)
        row3.addWidget(self.btn_load_md)

        self.btn_clear_glossary = QPushButton("✕")
        self.btn_clear_glossary.setToolTip("読み込み済みの単語帳辞書をクリア")
        self.btn_clear_glossary.setFixedWidth(32)
        self.btn_clear_glossary.clicked.connect(self._clear_glossary)
        row3.addWidget(self.btn_clear_glossary)

        control_layout.addLayout(row3)

        # 4行目: 手動の追加専門用語ヒント（任意）
        row4 = QHBoxLayout()
        row4.setSpacing(8)
        row4.addWidget(QLabel("追加ヒント:"))
        self.prompt_input = QLineEdit()
        self.prompt_input.setText(self.config.custom_prompt)
        self.prompt_input.setPlaceholderText(
            "任意: 特に重視したい固有名詞や講義テーマを手動追記できます（MD/動画ファイルは画面へドラッグ＆ドロップ可能）"
        )
        self.prompt_input.textChanged.connect(self._on_settings_changed)
        row4.addWidget(self.prompt_input, stretch=1)

        self.btn_clear_prompt = QPushButton("✕")
        self.btn_clear_prompt.setToolTip("手動ヒントをクリア")
        self.btn_clear_prompt.setFixedWidth(32)
        self.btn_clear_prompt.clicked.connect(lambda: self.prompt_input.clear())
        row4.addWidget(self.btn_clear_prompt)

        control_layout.addLayout(row4)

        # 動画・音声ファイル直接文字起こし用の進捗バー（処理中のみ表示）
        self.file_progress_bar = QProgressBar()
        self.file_progress_bar.setRange(0, 100)
        self.file_progress_bar.setValue(0)
        self.file_progress_bar.setTextVisible(True)
        self.file_progress_bar.setFormat("ファイル処理進捗: %p%")
        self.file_progress_bar.setFixedHeight(18)
        self.file_progress_bar.setVisible(False)
        control_layout.addWidget(self.file_progress_bar)

        main_layout.addWidget(control_card)

        # macOS向けシステム音声キャプチャ案内バナー（仮想オーディオ未検出時に表示）
        self.mac_hint_banner = QLabel(
            "💡 Macで講義動画（PC内部音声）を文字起こしするには、無料の仮想オーディオドライバ "
            "「BlackHole 2ch」を導入し、システム設定で複数出力装置を作成して上で選択してください。"
        )
        self.mac_hint_banner.setObjectName("HintBanner")
        self.mac_hint_banner.setWordWrap(True)
        self.mac_hint_banner.setVisible(False)
        main_layout.addWidget(self.mac_hint_banner)

        # --- ツールバー（検索・自動スクロール・コピー・保存・クリア） ---
        tool_row = QHBoxLayout()
        tool_row.setSpacing(8)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 文字起こし履歴を検索...")
        self.search_input.textChanged.connect(self._render_transcript_log)
        tool_row.addWidget(self.search_input, stretch=1)

        self.chk_auto_scroll = QCheckBox("自動スクロール")
        self.chk_auto_scroll.setChecked(self.config.auto_scroll)
        self.chk_auto_scroll.toggled.connect(self._on_settings_changed)
        tool_row.addWidget(self.chk_auto_scroll)

        self.btn_copy = QPushButton("📋 全文コピー")
        self.btn_copy.clicked.connect(self._copy_all_text)
        tool_row.addWidget(self.btn_copy)

        self.btn_export = QPushButton("💾 保存 (TXT/MD/SRT)")
        self.btn_export.clicked.connect(self._export_file)
        tool_row.addWidget(self.btn_export)

        self.btn_clear = QPushButton("🗑 クリア")
        self.btn_clear.clicked.connect(self._clear_transcripts)
        tool_row.addWidget(self.btn_clear)

        main_layout.addLayout(tool_row)

        # --- メイン文字起こしログ ---
        self.transcript_view = QTextEdit()
        self.transcript_view.setReadOnly(True)
        self.transcript_view.setPlaceholderText(
            "ここに講義の文字起こし結果がリアルタイムで表示されます。\n\n"
            "使い方:\n"
            "1. 上の「音声ソース」で [システム音声] を選択（WindowsはPCの再生音をそのまま取得できます）\n"
            "2. 「▶ 文字起こし開始」をクリックして講義動画を再生\n"
            "3. 必要に応じて「💬 字幕オーバーレイ」で動画の上に字幕を表示できます"
        )
        main_layout.addWidget(self.transcript_view, stretch=1)

        # --- リアルタイム暫定テキスト表示ボックス ---
        partial_card = QFrame()
        partial_card.setObjectName("PartialCard")
        partial_layout = QHBoxLayout(partial_card)
        partial_layout.setContentsMargins(12, 8, 12, 8)
        partial_layout.setSpacing(8)

        badge = QLabel("認識中")
        badge.setObjectName("LiveBadge")
        partial_layout.addWidget(badge)

        self.partial_label = QLabel("待機中...")
        self.partial_label.setObjectName("PartialText")
        self.partial_label.setWordWrap(True)
        partial_layout.addWidget(self.partial_label, stretch=1)

        main_layout.addWidget(partial_card)

        # --- ステータスバー ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("準備完了")
        self.stats_label = QLabel("00:00:00 | 0 セグメント")
        self.status_bar.addWidget(self.status_label, stretch=1)
        self.status_bar.addPermanentWidget(self.stats_label)

    def _apply_dark_theme(self) -> None:
        """見やすいモダンなダークテーマのスタイルシートを適用する"""
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #0F172A;
                color: #E2E8F0;
                font-family: "Yu Gothic UI", "Hiragino Sans", "Segoe UI", sans-serif;
                font-size: 13px;
            }
            QFrame#Card {
                background-color: #1E293B;
                border: 1px solid #334155;
                border-radius: 10px;
            }
            QFrame#PartialCard {
                background-color: #172554;
                border: 1px solid #1D4ED8;
                border-radius: 8px;
            }
            QLabel#LiveBadge {
                background-color: #2563EB;
                color: #FFFFFF;
                font-weight: bold;
                font-size: 11px;
                padding: 2px 8px;
                border-radius: 4px;
            }
            QLabel#PartialText {
                color: #BFDBFE;
                font-size: 14px;
            }
            QLabel#HintBanner {
                background-color: #1E293B;
                border: 1px solid #F59E0B;
                border-radius: 8px;
                color: #FDE68A;
                padding: 8px 12px;
                font-size: 12px;
            }
            QComboBox, QLineEdit {
                background-color: #0F172A;
                border: 1px solid #475569;
                border-radius: 6px;
                padding: 6px 10px;
                color: #F8FAFC;
            }
            QComboBox:hover, QLineEdit:hover {
                border-color: #60A5FA;
            }
            QPushButton {
                background-color: #334155;
                border: 1px solid #475569;
                border-radius: 6px;
                padding: 6px 12px;
                color: #F1F5F9;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #475569;
            }
            QPushButton:checked {
                background-color: #1D4ED8;
                border-color: #3B82F6;
            }
            QPushButton#PrimaryButton {
                background-color: #2563EB;
                border: 1px solid #3B82F6;
                color: #FFFFFF;
                font-weight: bold;
                padding: 7px 16px;
            }
            QPushButton#PrimaryButton:hover {
                background-color: #1D4ED8;
            }
            QPushButton#PrimaryButton[recording="true"] {
                background-color: #DC2626;
                border: 1px solid #EF4444;
            }
            QPushButton#PrimaryButton[recording="true"]:hover {
                background-color: #B91C1C;
            }
            QTextEdit {
                background-color: #0B1120;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                line-height: 1.6;
                color: #F1F5F9;
            }
            QProgressBar {
                background-color: #0F172A;
                border: 1px solid #334155;
                border-radius: 6px;
            }
            QProgressBar::chunk {
                background-color: #22C55E;
                border-radius: 5px;
            }
            QStatusBar {
                background-color: #0B1120;
                color: #94A3B8;
                border-top: 1px solid #1E293B;
            }
            """
        )

    @staticmethod
    def _select_combo_by_data(combo: QComboBox, target_data: str) -> None:
        """コンボボックスの内部データ値で項目を選択する"""
        for i in range(combo.count()):
            if combo.itemData(i) == target_data:
                combo.setCurrentIndex(i)
                return

    def _refresh_devices(self) -> None:
        """音声デバイスを再スキャンしてコンボボックスを更新する"""
        self.devices = list_audio_devices()
        self.device_combo.blockSignals(True)
        self.device_combo.clear()

        selected_idx = 0
        has_loopback = False
        for i, dev in enumerate(self.devices):
            self.device_combo.addItem(dev.name, dev.id)
            if dev.is_loopback:
                has_loopback = True
            if self.config.last_device_id and dev.id == self.config.last_device_id:
                selected_idx = i

        if self.devices:
            self.device_combo.setCurrentIndex(selected_idx)
        else:
            self.device_combo.addItem("利用可能な音声デバイスが見つかりません", "")

        self.device_combo.blockSignals(False)

        # Mac環境で仮想ループバックが見つからない場合は導入ガイドバナーを表示
        self.mac_hint_banner.setVisible(IS_MAC and not has_loopback)

    def _on_device_changed(self) -> None:
        """デバイス変更時に設定を保存する"""
        dev_id = self.device_combo.currentData()
        if isinstance(dev_id, str) and dev_id:
            self.config.last_device_id = dev_id
            self.config.save()

    def _on_settings_changed(self) -> None:
        """モデル・言語・専門用語ヒント・セクション絞り込み・自動スクロール設定を保存する"""
        self.config.model_size = str(self.model_combo.currentData() or "small")
        self.config.language = str(self.lang_combo.currentData() or "ja")
        self.config.custom_prompt = self.prompt_input.text().strip()
        self.config.glossary_section = str(self.section_combo.currentData() or "")
        self.config.auto_scroll = self.chk_auto_scroll.isChecked()
        self.config.save()

    def _get_selected_device(self) -> AudioDeviceInfo | None:
        """現在選択されている音声デバイス情報を返す"""
        idx = self.device_combo.currentIndex()
        if 0 <= idx < len(self.devices):
            return self.devices[idx]
        return None

    def _set_controls_busy(self, busy: bool, stop_label: str = "■ 停止") -> None:
        """録音中またはファイル処理中のボタン有効・無効状態を一括制御する"""
        self.btn_pause.setEnabled(busy)
        if not busy:
            self._is_paused = False
            self.btn_pause.setText("⏸ 一時停止")

        self.btn_start_stop.setText(stop_label if busy else "▶ 文字起こし開始")
        self.btn_start_stop.setProperty("recording", "true" if busy else "false")
        self.btn_start_stop.style().unpolish(self.btn_start_stop)
        self.btn_start_stop.style().polish(self.btn_start_stop)

        self.device_combo.setEnabled(not busy)
        self.model_combo.setEnabled(not busy)
        self.lang_combo.setEnabled(not busy)
        self.btn_refresh_dev.setEnabled(not busy)
        self.btn_file_transcribe.setEnabled(not busy)

    def _toggle_recording(self) -> None:
        """リアルタイム文字起こしの開始・停止（またはファイル処理の中断）を切り替える"""
        if self._is_file_running:
            self._stop_file_transcription()
        elif self._is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _toggle_pause(self) -> None:
        """リアルタイム録音または動画ファイル文字起こしの一時停止・再開を切り替える"""
        if not (self._is_recording or self._is_file_running):
            return
        self._is_paused = not self._is_paused

        if self._is_recording:
            if self._is_paused:
                self._paused_at = time.monotonic()
            elif self._paused_at is not None:
                self._record_start_time += time.monotonic() - self._paused_at
                self._paused_at = None
            if self._capture_stream is not None:
                self._capture_stream.set_paused(self._is_paused)
            if self._transcriber is not None:
                self._transcriber.set_paused(self._is_paused)
            if self._is_paused:
                self._current_level = 0.0
                self.level_bar.setValue(0)
                self.partial_label.setText("一時停止中（▶ 再開 で続きから文字起こしできます）")

        if self._is_file_running and self._file_transcriber is not None:
            self._file_transcriber.set_paused(self._is_paused)
            if self._is_paused:
                self.partial_label.setText("ファイル文字起こしを一時停止中...")

        self.btn_pause.setText("▶ 再開" if self._is_paused else "⏸ 一時停止")

    def _start_recording(self) -> None:
        """音声キャプチャとリアルタイム文字起こしワーカーを開始する"""
        device = self._get_selected_device()
        if device is None:
            QMessageBox.warning(
                self,
                "デバイス未選択",
                "有効な音声ソースが選択されていません。デバイス一覧を更新してください。",
            )
            return

        # キューを空にして新しいセッションを開始
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        self._on_settings_changed()
        self._record_start_time = time.monotonic()
        self._paused_at = None
        self._is_paused = False

        self._transcriber = RealtimeTranscriber(
            config=self.config,
            audio_queue=self._audio_queue,
            glossary=self.glossary,
            on_partial=self._bridge.partial_received.emit,
            on_final=self._bridge.final_received.emit,
            on_status=self._bridge.status_received.emit,
            on_error=self._bridge.error_received.emit,
        )
        self._capture_stream = AudioCaptureStream(
            device=device,
            audio_queue=self._audio_queue,
            level_callback=self._bridge.level_received.emit,
        )

        try:
            self._transcriber.start(initial_segment_index=len(self.segments))
            self._capture_stream.start()
        except Exception as exc:
            self._stop_recording()
            QMessageBox.critical(
                self,
                "録音開始エラー",
                f"音声キャプチャの開始に失敗しました:\n{exc}",
            )
            return

        self._is_recording = True
        self._set_controls_busy(True, stop_label="■ 停止")

    def _stop_recording(self) -> None:
        """音声キャプチャとリアルタイム文字起こしワーカーを停止する"""
        if self._capture_stream is not None:
            self._capture_stream.stop()
            self._capture_stream = None

        if self._transcriber is not None:
            self._transcriber.stop()
            self._transcriber = None

        self._is_recording = False
        self._is_paused = False
        self._paused_at = None
        self._current_level = 0.0
        self.level_bar.setValue(0)
        self.partial_label.setText("停止中")
        self._set_controls_busy(False)
        self.status_label.setText("停止しました")

    def _open_media_file_dialog(self) -> None:
        """直接文字起こしする動画・音声ファイルを選択するダイアログを開く"""
        exts = " ".join(f"*{e}" for e in SUPPORTED_MEDIA_EXTENSIONS)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "文字起こしする動画・音声ファイルを選択",
            "",
            f"動画・音声ファイル ({exts});;すべてのファイル (*.*)",
        )
        if file_path:
            self._start_file_transcription(Path(file_path))

    def _start_file_transcription(self, file_path: Path) -> None:
        """選択された動画・音声ファイルの直接文字起こしを開始する"""
        if self._is_recording or self._is_file_running:
            QMessageBox.information(
                self,
                "処理中",
                "現在進行中の文字起こしを停止してから動画ファイルを読み込んでください。",
            )
            return

        if not file_path.is_file():
            QMessageBox.warning(self, "ファイルエラー", f"ファイルが見つかりません:\n{file_path}")
            return

        self._on_settings_changed()
        self._is_file_running = True
        self._is_paused = False
        self.file_progress_bar.setValue(0)
        self.file_progress_bar.setFormat(f"準備中: {file_path.name} (%p%)")
        self.file_progress_bar.setVisible(True)
        self._set_controls_busy(True, stop_label="■ ファイル処理中断")

        self._file_transcriber = FileBatchTranscriber(
            file_path=file_path,
            config=self.config,
            glossary=self.glossary,
            on_segment=self._bridge.final_received.emit,
            on_progress=self._bridge.file_progress_received.emit,
            on_status=self._bridge.status_received.emit,
            on_finished=self._bridge.file_finished_received.emit,
            on_error=self._bridge.error_received.emit,
        )
        self._file_transcriber.start(initial_segment_index=len(self.segments))

    def _stop_file_transcription(self) -> None:
        """実行中の動画・音声ファイル文字起こしを中断する"""
        if self._file_transcriber is not None:
            self._file_transcriber.stop()
            self._file_transcriber = None
        self._is_file_running = False
        self.file_progress_bar.setVisible(False)
        self.partial_label.setText("ファイル処理を中断しました")
        self._set_controls_busy(False)

    def _on_file_progress(self, current_sec: float, total_sec: float) -> None:
        """ファイル文字起こしの進捗バーを更新する"""
        if total_sec > 0:
            pct = min(100, int((current_sec / total_sec) * 100))
            self.file_progress_bar.setValue(pct)
            self.file_progress_bar.setFormat(
                f"文字起こし中: %p% ({format_hhmmss(current_sec)} / {format_hhmmss(total_sec)})"
            )
        else:
            self.file_progress_bar.setFormat(
                f"文字起こし中: {format_hhmmss(current_sec)} 処理済"
            )

    def _on_file_finished(self, completed: bool) -> None:
        """ファイル文字起こし完了時の後処理"""
        self._file_transcriber = None
        self._is_file_running = False
        self._set_controls_busy(False)
        if completed:
            self.file_progress_bar.setValue(100)
            self.file_progress_bar.setFormat("文字起こし完了 (100%)")
            self.partial_label.setText("ファイル文字起こしが完了しました。「💾 保存」でノートを出力できます。")
        else:
            self.file_progress_bar.setVisible(False)

    def _toggle_overlay(self, checked: bool) -> None:
        """フローティング字幕オーバーレイウィンドウの表示・非表示を切り替える"""
        if checked:
            if self._overlay is None:
                self._overlay = SubtitleOverlayWindow(
                    font_size=self.config.overlay_font_size,
                    opacity=self.config.overlay_opacity,
                )
                self._overlay.closed.connect(self._on_overlay_closed)
                self._overlay.font_size_changed.connect(self._on_overlay_font_changed)
            self._overlay.show()
            self._overlay.raise_()
        else:
            if self._overlay is not None:
                self._overlay.close()

    def _on_overlay_closed(self) -> None:
        """字幕ウィンドウが閉じられたときにトグルボタンを戻す"""
        self.btn_overlay.blockSignals(True)
        self.btn_overlay.setChecked(False)
        self.btn_overlay.blockSignals(False)
        self._overlay = None

    def _on_overlay_font_changed(self, size: int) -> None:
        """字幕ウィンドウのフォントサイズ変更を設定に保存する"""
        self.config.overlay_font_size = size
        self.config.save()

    def _on_partial_text(self, text: str) -> None:
        """暫定文字起こしテキストを受信して表示更新する"""
        self.partial_label.setText(text if text else "音声を聞いています...")
        if self._overlay is not None:
            self._overlay.set_partial_text(text)

    def _on_final_segment(self, segment: TranscriptSegment) -> None:
        """確定した文字起こしセグメントを追加してログを更新する"""
        self.segments.append(segment)
        self._render_transcript_log()
        if self._overlay is not None:
            self._overlay.set_final_text(segment.text)

    def _on_status_message(self, message: str) -> None:
        """ステータスバーのメッセージを更新する"""
        self.status_label.setText(message)
        # モデル読み込み中や切替時は認識中パネルにも状態を表示して分かりやすくする
        if "読み込み中" in message or "切り替え" in message:
            self.partial_label.setText(message)
        elif "リアルタイム文字起こし中" in message:
            self.partial_label.setText("音声を聞いています...")

    def _on_error_message(self, message: str) -> None:
        """エラー通知を受け取り処理中のワーカーを停止してダイアログ表示する"""
        if self._is_recording:
            self._stop_recording()
        if self._is_file_running:
            self._stop_file_transcription()
        QMessageBox.critical(self, "エラー", message)

    def _on_audio_level(self, rms: float) -> None:
        """音声RMSレベルを受け取り視覚的に見やすい0〜100スケールへ変換する"""
        # 対数スケール風に見やすく増幅
        scaled = min(100.0, rms * 900.0)
        self._current_level = max(scaled, self._current_level * 0.7)

    def _on_ui_tick(self) -> None:
        """100msごとに音量メーターと経過時間を更新する"""
        if self._is_recording:
            if self._is_paused:
                self.level_bar.setValue(0)
                elapsed = (
                    (self._paused_at or time.monotonic()) - self._record_start_time
                )
                self.stats_label.setText(
                    f"⏸ {format_hhmmss(elapsed)} | {len(self.segments)} セグメント"
                )
            else:
                self.level_bar.setValue(int(self._current_level))
                self._current_level *= 0.82
                elapsed = time.monotonic() - self._record_start_time
                self.stats_label.setText(
                    f"{format_hhmmss(elapsed)} | {len(self.segments)} セグメント"
                )
        else:
            self.stats_label.setText(f"{len(self.segments)} セグメント")

    def _render_transcript_log(self) -> None:
        """検索キーワードを考慮して文字起こしログを再描画する"""
        query = self.search_input.text().strip().lower()
        html_lines: list[str] = []

        for seg in self.segments:
            if query and query not in seg.text.lower():
                continue
            ts = format_hhmmss(seg.start_sec)
            escaped = (
                seg.text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            html_lines.append(
                f"<div style='margin-bottom:6px;'>"
                f"<span style='color:#60A5FA; font-family:monospace;'>[{ts}]</span> "
                f"<span style='color:#F1F5F9;'>{escaped}</span>"
                f"</div>"
            )

        self.transcript_view.setHtml("".join(html_lines))
        if self.config.auto_scroll and not query:
            cursor = self.transcript_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.transcript_view.setTextCursor(cursor)

    def _copy_all_text(self) -> None:
        """すべての文字起こしテキストをクリップボードへコピーする"""
        if not self.segments:
            return
        full_text = "\n".join(
            f"[{format_hhmmss(s.start_sec)}] {s.text}" for s in self.segments
        )
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(full_text)
            self.status_label.setText("文字起こし全文をクリップボードにコピーしました")

    def _export_file(self) -> None:
        """文字起こし結果をTXT/MD/SRTファイルとして保存する"""
        if not self.segments:
            QMessageBox.information(
                self, "保存", "保存する文字起こしデータがまだありません。"
            )
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "文字起こし結果を保存",
            "lecture_transcript.txt",
            "テキストファイル (*.txt);;Markdownノート (*.md);;SRT字幕ファイル (*.srt)",
        )
        if not file_path:
            return

        fmt = "txt"
        if "Markdown" in selected_filter or file_path.lower().endswith(".md"):
            fmt = "md"
        elif "SRT" in selected_filter or file_path.lower().endswith(".srt"):
            fmt = "srt"

        saved_path = export_segments(self.segments, file_path, fmt=fmt)
        self.status_label.setText(f"保存しました: {saved_path}")

    def _clear_transcripts(self) -> None:
        """記録済みの文字起こしログをクリアする"""
        if not self.segments:
            return
        reply = QMessageBox.question(
            self,
            "履歴のクリア",
            "現在の文字起こし履歴をすべて消去しますか？",
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.segments.clear()
            self.transcript_view.clear()
            if self._overlay is not None:
                self._overlay.clear_text()
            self.status_label.setText("履歴をクリアしました")

    def _load_glossary_dialog(self) -> None:
        """ファイル選択ダイアログから単語帳MD/TXTファイルを選び全単語を辞書登録する"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "全単語を登録する単語帳・講義資料を選択",
            "",
            "Markdown / テキスト (*.md *.markdown *.txt);;すべてのファイル (*.*)",
        )
        if file_paths:
            self._load_glossary_from_paths([Path(p) for p in file_paths])

    def _load_glossary_from_paths(self, paths: list[Path]) -> None:
        """指定されたファイル群を単語帳辞書として設定へ保存し全単語をロードする"""
        valid_paths = [str(p) for p in paths if p.is_file()]
        if not valid_paths:
            return
        self.config.glossary_files = valid_paths
        self.config.glossary_section = ""
        self.config.save()
        self._reload_glossary_dictionary(silent=False)

    def _reload_glossary_dictionary(self, silent: bool = False) -> None:
        """設定に保存された単語帳ファイル群から全単語辞書を構築しUIへ反映する"""
        existing_paths = [p for p in self.config.glossary_files if Path(p).is_file()]
        if not existing_paths:
            self.glossary = GlossaryDictionary()
            self.glossary_status_label.setText(
                "未読込（MD/TXTファイルを読み込むと全単語を自動認識・補正します）"
            )
            self.glossary_status_label.setStyleSheet("color: #94A3B8;")
            self.section_combo.blockSignals(True)
            self.section_combo.clear()
            self.section_combo.addItem("すべて (会話から自動追従)", "")
            self.section_combo.blockSignals(False)
            return

        self.glossary = GlossaryDictionary.from_files(existing_paths)
        names = ", ".join(Path(p).name for p in existing_paths)
        self.glossary_status_label.setText(
            f"📚 {names}（全 {self.glossary.total_terms:,} 語 / {len(self.glossary.sections)} セクション）"
        )
        self.glossary_status_label.setStyleSheet("color: #4ADE80; font-weight: 600;")

        # セクション絞り込みコンボボックスを更新
        self.section_combo.blockSignals(True)
        self.section_combo.clear()
        self.section_combo.addItem("すべて (会話から自動追従)", "")
        for sec in self.glossary.sections:
            count = len(self.glossary.section_to_terms.get(sec, []))
            self.section_combo.addItem(f"{sec} ({count}語)", sec)
        self._select_combo_by_data(self.section_combo, self.config.glossary_section)
        self.section_combo.blockSignals(False)

        if not silent:
            self.status_label.setText(
                f"単語帳を読み込みました: {names}（全 {self.glossary.total_terms:,} 語）"
            )

    def _clear_glossary(self) -> None:
        """読み込み済みの単語帳辞書をクリアする"""
        self.config.glossary_files = []
        self.config.glossary_section = ""
        self.config.save()
        self._reload_glossary_dictionary(silent=True)
        self.status_label.setText("単語帳辞書をクリアしました")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """MD/TXTファイルおよび動画・音声ファイルのドラッグを受け入れる"""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if not url.isLocalFile():
                    continue
                lower = url.toLocalFile().lower()
                if lower.endswith((".md", ".markdown", ".txt")) or lower.endswith(
                    SUPPORTED_MEDIA_EXTENSIONS
                ):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        """ドロップされたファイルに応じて単語帳読込または動画・音声文字起こしを実行する"""
        md_paths: list[Path] = []
        media_paths: list[Path] = []

        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            p = Path(url.toLocalFile())
            lower = p.name.lower()
            if lower.endswith((".md", ".markdown", ".txt")):
                md_paths.append(p)
            elif lower.endswith(SUPPORTED_MEDIA_EXTENSIONS):
                media_paths.append(p)

        if md_paths:
            self._load_glossary_from_paths(md_paths)
            event.acceptProposedAction()
        if media_paths:
            self._start_file_transcription(media_paths[0])
            event.acceptProposedAction()

    def closeEvent(self, event) -> None:
        """アプリ終了時に録音・ファイル処理スレッドと字幕ウィンドウを安全に終了する"""
        self._stop_recording()
        if self._is_file_running:
            self._stop_file_transcription()
        if self._overlay is not None:
            self._overlay.close()
        super().closeEvent(event)
