# -*- mode: python ; coding: utf-8 -*-
"""macOS用 LectureTranscriber.app ビルド定義ファイル (PyInstaller Spec)

Retinaディスプレイ対応、マイク／仮想オーディオ入力権限 (NSMicrophoneUsageDescription)、
および faster-whisper / ctranslate2 / PyAV / sounddevice 等のネイティブライブラリを
漏れなくバンドルしてスタンドアロン .app を生成する。
"""

from PyInstaller.utils.hooks import collect_all, collect_data_files

# ネイティブライブラリやデータファイルを一括収集するパッケージ
packages_to_collect = [
    "faster_whisper",
    "ctranslate2",
    "onnxruntime",
    "av",
    "sounddevice",
    "_sounddevice_data",
]

datas = []
binaries = []
hiddenimports = [
    "src.audio_capture",
    "src.config",
    "src.exporter",
    "src.file_transcriber",
    "src.glossary",
    "src.main_window",
    "src.subtitle_overlay",
    "src.transcriber",
]

for pkg in packages_to_collect:
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hidden
    except Exception:
        # 環境によって存在しない補助パッケージは無視して続行
        pass

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PIL.ImageTk"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LectureTranscriber",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="LectureTranscriber",
)

# macOS アプリケーションバンドル (.app) の定義
app = BUNDLE(
    coll,
    name="LectureTranscriber.app",
    icon=None,
    bundle_identifier="com.local.lecturetranscriber",
    info_plist={
        "CFBundleName": "Lecture Transcriber",
        "CFBundleDisplayName": "Lecture Transcriber",
        "CFBundleShortVersionString": "1.0.0",
        "CFBundleVersion": "1.0.0",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "12.0",
        # macOSでマイク・BlackHole等の音声入力を許可するための必須説明文
        "NSMicrophoneUsageDescription": (
            "講義動画やマイクの音声をリアルタイムで文字起こしするために音声入力を使用します。"
        ),
        # ドラッグ＆ドロップで開けるファイル種別（MD/TXT・動画/音声）
        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeName": "Media and Glossary Files",
                "CFBundleTypeRole": "Viewer",
                "LSItemContentTypes": [
                    "public.movie",
                    "public.audio",
                    "public.plain-text",
                    "net.daringfireball.markdown",
                ],
            }
        ],
    },
)
