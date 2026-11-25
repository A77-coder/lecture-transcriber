# -*- coding: utf-8 -*-
"""講義動画の上に重ねて表示できるフローティング字幕オーバーレイウィンドウ"""

from __future__ import annotations

from PySide6.QtCore import Qt, QPoint, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizeGrip,
    QVBoxLayout,
    QWidget,
)


class SubtitleOverlayWindow(QWidget):
    """常に最前面に表示される半透明のフローティング字幕ウィンドウ"""

    closed = Signal()  # ユーザーによって閉じられたときに発火するシグナル
    font_size_changed = Signal(int)  # フォントサイズ変更通知

    def __init__(self, font_size: int = 20, opacity: float = 0.82) -> None:
        super().__init__()
        self._font_size = font_size
        self._opacity = max(0.3, min(1.0, opacity))
        self._drag_pos: QPoint | None = None
        self._final_lines: list[str] = []
        self._partial_text: str = ""

        # フレームレスかつ常に最前面に表示するウィンドウ設定
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowTitle("リアルタイム字幕")
        self.resize(820, 140)
        self.setMinimumSize(360, 90)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """オーバーレイ内のレイアウトとコントロールを構築する"""
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 10, 16, 10)
        root_layout.setSpacing(6)

        # 上部ツールバー（ドラッグ案内・フォントサイズ変更・閉じるボタン）
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)

        hint_label = QLabel("⠿ ドラッグで移動")
        hint_label.setStyleSheet("color: rgba(255, 255, 255, 130); font-size: 11px;")
        top_bar.addWidget(hint_label)
        top_bar.addStretch()

        btn_style = (
            "QPushButton {"
            "  background-color: rgba(255, 255, 255, 28);"
            "  color: #FFFFFF;"
            "  border: none;"
            "  border-radius: 4px;"
            "  padding: 2px 8px;"
            "  font-size: 12px;"
            "}"
            "QPushButton:hover { background-color: rgba(255, 255, 255, 55); }"
        )

        btn_smaller = QPushButton("A-")
        btn_smaller.setToolTip("文字を小さくする")
        btn_smaller.setStyleSheet(btn_style)
        btn_smaller.clicked.connect(lambda: self.adjust_font_size(-2))
        top_bar.addWidget(btn_smaller)

        btn_larger = QPushButton("A+")
        btn_larger.setToolTip("文字を大きくする")
        btn_larger.setStyleSheet(btn_style)
        btn_larger.clicked.connect(lambda: self.adjust_font_size(2))
        top_bar.addWidget(btn_larger)

        btn_close = QPushButton("✕")
        btn_close.setToolTip("字幕ウィンドウを閉じる")
        btn_close.setStyleSheet(btn_style)
        btn_close.clicked.connect(self.close)
        top_bar.addWidget(btn_close)

        root_layout.addLayout(top_bar)

        # 字幕表示ラベル
        self.subtitle_label = QLabel("音声待機中...")
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._apply_label_style()
        root_layout.addWidget(self.subtitle_label, stretch=1)

        # 右下のリサイズグリップ
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(0, 0, 0, 0)
        bottom_bar.addStretch()
        grip = QSizeGrip(self)
        grip.setStyleSheet("background: transparent;")
        bottom_bar.addWidget(grip)
        root_layout.addLayout(bottom_bar)

    def _apply_label_style(self) -> None:
        """現在のフォントサイズに応じたスタイルを字幕ラベルへ適用する"""
        self.subtitle_label.setStyleSheet(
            f"color: #FFFFFF; font-size: {self._font_size}px; font-weight: 600;"
            "background: transparent;"
        )

    def adjust_font_size(self, delta: int) -> None:
        """字幕フォントサイズを上下させる"""
        self._font_size = max(12, min(42, self._font_size + delta))
        self._apply_label_style()
        self.font_size_changed.emit(self._font_size)

    def set_final_text(self, text: str) -> None:
        """確定テキストを追加し、最新2行を表示する"""
        if not text:
            return
        self._final_lines.append(text)
        if len(self._final_lines) > 2:
            self._final_lines = self._final_lines[-2:]
        self._partial_text = ""
        self._refresh_display()

    def set_partial_text(self, text: str) -> None:
        """認識中の暫定テキストを更新する"""
        self._partial_text = text
        self._refresh_display()

    def clear_text(self) -> None:
        """字幕表示をクリアする"""
        self._final_lines.clear()
        self._partial_text = ""
        self.subtitle_label.setText("音声待機中...")

    def _refresh_display(self) -> None:
        """確定行と暫定行をリッチテキストで結合表示する"""
        parts: list[str] = []
        for line in self._final_lines:
            escaped = (
                line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )
            parts.append(f"<span style='color:#FFFFFF;'>{escaped}</span>")
        if self._partial_text:
            escaped_p = (
                self._partial_text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            parts.append(f"<span style='color:#93C5FD;'>{escaped_p}</span>")

        if not parts:
            self.subtitle_label.setText("音声待機中...")
        else:
            self.subtitle_label.setText("<br>".join(parts))

    def paintEvent(self, event: QPaintEvent) -> None:
        """角丸・半透明のダーク背景を描画する"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        alpha = int(self._opacity * 255)
        painter.setBrush(QColor(15, 23, 42, alpha))
        painter.setPen(QColor(148, 163, 184, 90))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 14, 14)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """ドラッグ移動開始位置を記録する"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """ウィンドウをドラッグ移動する"""
        if (
            self._drag_pos is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """ドラッグ終了"""
        self._drag_pos = None

    def closeEvent(self, event) -> None:
        """ウィンドウが閉じられたことを親へ通知する"""
        self.closed.emit()
        super().closeEvent(event)
