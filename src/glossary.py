# -*- coding: utf-8 -*-
"""Markdown / テキストファイルから全専門用語を構造化読み込みし、
動的プロンプト生成および文字起こし結果の表記自動補正を行うモジュール
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

# 一般的な語句やMarkdownノイズを除外するためのストップワード
_STOP_WORDS = {
    "こと",
    "もの",
    "ため",
    "これ",
    "それ",
    "あれ",
    "について",
    "まとめ",
    "概要",
    "目次",
    "参考",
    "参考文献",
    "例題",
    "演習",
    "問題",
    "解答",
    "注意",
    "補足",
    "http",
    "https",
    "www",
    "com",
    "org",
    "png",
    "jpg",
    "jpeg",
    "pdf",
    "true",
    "false",
    "none",
    "null",
}

_URL_RE = re.compile(r"https?://\S+")
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+)$")
_BOLD_BULLET_RE = re.compile(r"^\s*[-*+]\s+\*\*([^*]{1,60})\*\*")
_BOLD_RE = re.compile(r"\*\*([^*]{2,40})\*\*")
_INLINE_CODE_RE = re.compile(r"`([^`\n]{2,40})`")
_BULLET_TERM_RE = re.compile(r"^\s*[-*+]\s+([^:：\n]{2,30})[:：]")

_KATAKANA_RE = re.compile(r"[ァ-ヴー]{3,24}")
_KATAKANA_SEQ_RE = re.compile(r"[ァ-ヴー・＝=]{4,28}")
_ALPHA_TERM_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_+-]{1,24}\b")
_KANJI_TERM_RE = re.compile(r"[一-龥]{3,10}")
_SEP_NORMALIZE_RE = re.compile(r"[・＝=\s　]+")


def _clean_term(term: str) -> str:
    """抽出した候補語句から余分なMarkdown記号を取り除く（固有名詞の＝や・は保持）"""
    cleaned = re.sub(r"[#*`\[\]()（）「」『』【】<>:;,、。\\|]", " ", term)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _is_valid_term(term: str) -> bool:
    """専門用語として有効な語句かを判定する"""
    if len(term) < 2 or len(term) > 36:
        return False
    if term.lower() in _STOP_WORDS:
        return False
    if re.fullmatch(r"[0-9.\-_]+", term):
        return False
    return True


def _normalize_key(term: str) -> str:
    """記号（＝・スペース）を除去した照合用正規化キーを生成する"""
    return _SEP_NORMALIZE_RE.sub("", term).lower()


@dataclass
class GlossaryEntry:
    """辞書内の1単語エントリ"""

    term: str  # 正式表記（例: ホモ＝サピエンス）
    section: str  # 所属セクション（例: 序章 先史の世界）
    norm_key: str  # 記号除去後の照合キー


@dataclass
class GlossaryDictionary:
    """大規模単語帳（数千語以上）の全単語を保持し、動的プロンプト生成と表記補正を行う辞書"""

    entries: list[GlossaryEntry] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)
    section_to_terms: dict[str, list[str]] = field(default_factory=dict)
    # 記号ゆれ完全一致用マップ（正規化キー -> 正式表記）
    _exact_norm_map: dict[str, str] = field(default_factory=dict)
    # カタカナ語ファジー補正用リスト（正規化カタカナ -> 正式表記）
    _katakana_map: dict[str, str] = field(default_factory=dict)
    # 単語 -> 所属セクションの逆引き（自動セクション追従用）
    _term_to_section: dict[str, str] = field(default_factory=dict)
    _katakana_keys_by_len: dict[int, list[str]] = field(default_factory=dict)
    _active_section: str = ""  # 自動推定された現在のアクティブセクション

    @property
    def total_terms(self) -> int:
        """辞書に登録されている全単語数を返す"""
        return len(self.entries)

    def add_term(self, raw_term: str, section: str) -> None:
        """1つの単語（スラッシュ区切りの別名を含む場合あり）を辞書に登録する"""
        # 例: "ホモ＝エレクトゥス／原人" -> 両方を個別に登録
        parts = re.split(r"[／/]", raw_term)
        for part in parts:
            term = _clean_term(part)
            if not _is_valid_term(term):
                continue
            norm = _normalize_key(term)
            if not norm or norm in self._exact_norm_map:
                continue

            entry = GlossaryEntry(term=term, section=section, norm_key=norm)
            self.entries.append(entry)
            self._exact_norm_map[norm] = term
            self._term_to_section[term] = section

            if section not in self.section_to_terms:
                self.section_to_terms[section] = []
                if section not in self.sections:
                    self.sections.append(section)
            self.section_to_terms[section].append(term)

            # 4文字以上の純カタカナ（記号含む）語はファジー補正インデックスにも登録
            if len(norm) >= 4 and re.fullmatch(r"[ァ-ヴー]+", norm):
                self._katakana_map[norm] = term
                self._katakana_keys_by_len.setdefault(len(norm), []).append(norm)

    @classmethod
    def from_markdown_text(cls, content: str, default_section: str = "一般") -> "GlossaryDictionary":
        """Markdown/テキスト本文から全単語を構造化して読み込む"""
        dictionary = cls()
        text = _CODE_BLOCK_RE.sub("\n", content)
        text = _URL_RE.sub(" ", text)
        text = _MD_LINK_RE.sub(r"\1", text)

        current_h2 = ""
        current_h3 = ""
        in_toc = False
        structured_count = 0

        for line in text.splitlines():
            h_match = _HEADING_RE.match(line)
            if h_match:
                level = len(h_match.group(1))
                title = h_match.group(2).strip()
                if title in ("目次", "Contents"):
                    in_toc = True
                    continue
                in_toc = False
                if level == 2:
                    current_h2 = title
                    current_h3 = ""
                elif level == 3:
                    current_h3 = title
                continue

            if in_toc:
                continue

            section_name = (
                f"{current_h2} / {current_h3}"
                if (current_h2 and current_h3)
                else (current_h2 or default_section)
            )

            # 単語帳形式: "- **単語名**：説明..." を最優先で全件抽出
            m_bold_bullet = _BOLD_BULLET_RE.match(line)
            if m_bold_bullet:
                dictionary.add_term(m_bold_bullet.group(1), section_name)
                structured_count += 1
                continue

            # 通常の箇条書き定義形式: "- 用語：説明"
            m_bullet = _BULLET_TERM_RE.match(line)
            if m_bullet:
                dictionary.add_term(m_bullet.group(1), section_name)
                structured_count += 1

            # 行中の太字・インラインコードも拾う
            for b in _BOLD_RE.findall(line):
                dictionary.add_term(b, section_name)
            for c in _INLINE_CODE_RE.findall(line):
                dictionary.add_term(c, section_name)

        # 構造化単語が見つからない自由記述テキストの場合のみ本文中の専門語を抽出
        if structured_count == 0:
            freq: dict[str, int] = {}
            for pattern in (_KATAKANA_RE, _ALPHA_TERM_RE, _KANJI_TERM_RE):
                for m in pattern.findall(text):
                    t = _clean_term(m)
                    if _is_valid_term(t):
                        freq[t] = freq.get(t, 0) + 1
            for term, _cnt in sorted(freq.items(), key=lambda kv: (-kv[1], -len(kv[0]))):
                dictionary.add_term(term, default_section)

        return dictionary

    @classmethod
    def from_files(cls, file_paths: list[str | Path]) -> "GlossaryDictionary":
        """複数のMD/TXTファイルを統合して1つの全単語辞書を構築する"""
        merged = cls()
        for fp in file_paths:
            path = Path(fp)
            if not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = path.read_text(encoding="cp932", errors="ignore")
            sub = cls.from_markdown_text(content, default_section=path.stem)
            for entry in sub.entries:
                merged.add_term(entry.term, entry.section)
        return merged

    def update_context_from_text(self, spoken_text: str) -> None:
        """直近の発話テキストから該当セクションを自動推定してアクティブセクションを更新する"""
        if not spoken_text or not self.entries:
            return
        norm_spoken = _normalize_key(spoken_text)
        section_hits: dict[str, int] = {}
        # 3文字以上の単語が発話に含まれていれば該当セクションにスコア加算
        for entry in self.entries:
            if len(entry.norm_key) >= 3 and entry.norm_key in norm_spoken:
                section_hits[entry.section] = section_hits.get(entry.section, 0) + 1
        if section_hits:
            best_sec = max(section_hits.items(), key=lambda kv: kv[1])[0]
            self._active_section = best_sec

    def build_dynamic_prompt(
        self,
        selected_section: str = "",
        recent_text: str = "",
        manual_hint: str = "",
        max_chars: int = 180,
    ) -> str | None:
        """Whisperのトークン上限内で最も効果的なinitial_promptを動的に生成する"""
        if recent_text:
            self.update_context_from_text(recent_text)

        parts: list[str] = []
        used_chars = 0

        # 1. 手動入力のヒントがあれば最優先
        manual = manual_hint.strip()
        if manual:
            snippet = manual[:100]
            parts.append(snippet)
            used_chars += len(snippet)

        # 2. 選択中セクション（または自動追従で推定されたセクション）の単語を流し込む
        target_sec = selected_section or self._active_section
        candidate_terms: list[str] = []
        if target_sec and target_sec in self.section_to_terms:
            candidate_terms = self.section_to_terms[target_sec]
        elif self.entries:
            candidate_terms = [e.term for e in self.entries[:40]]

        if candidate_terms and used_chars < max_chars - 20:
            picked: list[str] = []
            for term in candidate_terms:
                add_len = len(term) + (1 if picked else 0)
                if used_chars + add_len > max_chars - 50:
                    break
                picked.append(term)
                used_chars += add_len
            if picked:
                parts.append("、".join(picked))

        # 3. 直前の確定文の末尾を文脈として連結
        if recent_text:
            tail = recent_text.strip()[-55:]
            if tail:
                parts.append(tail)

        prompt = " ".join(parts).strip()
        return prompt if prompt else None

    def correct_text(self, text: str) -> str:
        """文字起こし結果に含まれる専門用語の表記揺れ・カタカナ誤認識を全単語辞書で補正する"""
        if not text or not self.entries:
            return text

        # 1. カタカナ列（記号含む）の表記揺れ・類似語補正
        def _replace_katakana(match: re.Match[str]) -> str:
            raw = match.group(0)
            norm = _normalize_key(raw)
            if len(norm) < 4:
                return raw

            # 記号違いの完全一致（例: "ホモサピエンス" -> "ホモ＝サピエンス"）
            if norm in self._katakana_map:
                return self._katakana_map[norm]

            # 長さが±1以内の辞書カタカナ語から高精度ファジーマッチ（誤変換防止のため0.84以上）
            candidates: list[str] = []
            for length in (len(norm) - 1, len(norm), len(norm) + 1):
                candidates.extend(self._katakana_keys_by_len.get(length, []))
            if not candidates:
                return raw

            best_matches = difflib.get_close_matches(norm, candidates, n=1, cutoff=0.84)
            if best_matches:
                return self._katakana_map[best_matches[0]]
            return raw

        corrected = _KATAKANA_SEQ_RE.sub(_replace_katakana, text)
        return corrected


# 後方互換用のヘルパー関数
def extract_keywords_from_markdown(
    content: str,
    max_terms: int = 35,
    max_chars: int = 180,
) -> list[str]:
    """Markdownテキストから代表的なキーワードリストを返す（互換API）"""
    d = GlossaryDictionary.from_markdown_text(content)
    selected: list[str] = []
    total = 0
    for entry in d.entries:
        if len(selected) >= max_terms:
            break
        add_len = len(entry.term) + (2 if selected else 0)
        if total + add_len > max_chars:
            continue
        selected.append(entry.term)
        total += add_len
    return selected


def extract_keywords_from_file(file_path: str | Path) -> list[str]:
    """ファイルから代表的なキーワードリストを返す（互換API）"""
    d = GlossaryDictionary.from_files([file_path])
    return [e.term for e in d.entries[:35]]
