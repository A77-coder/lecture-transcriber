# -*- coding: utf-8 -*-
"""講義動画リアルタイム文字起こしアプリ エントリーポイント"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from src.main_window import MainWindow


def main() -> int:
    """アプリケーションを起動するメイン関数"""
    app = QApplication(sys.argv)
    app.setApplicationName("Lecture Transcriber")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
