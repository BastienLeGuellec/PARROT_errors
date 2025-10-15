#!/usr/bin/env python3
"""
Excel processor with per-row language and improved randomness:
- For each row, read the language code from `lang` and transform text in `report`.
- One random word per sentence is replaced by a phonetic neighbor using Double Metaphone.
- Combines candidates from both primary and secondary codes.
- Robust "same word" filtering (casefold, diacritics, inner punctuation).
- RNG is threaded through (no global reseeding), so seeds work as expected.

Dependencies:
    pip install Metaphone wordfreq regex pandas openpyxl

CLI examples:
    python dmeta_excel.py --excel reports.xlsx --out reports_dmeta.xlsx
    python dmeta_excel.py --excel reports.xlsx --seed 42 --in-place
    python dmeta_excel.py --excel reports.xlsx --report-col text --lang-col language --max-words-per-lang 100000
"""

from __future__ import annotations
import argparse
import random
import re
import unicodedata
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union

import pandas as pd

# Maps full language names to two-letter codes for wordfreq
LANG_NAME_TO_CODE = {
    'english': 'en',
    'french': 'fr',
    'spanish': 'es',
    'german': 'de',
    'italian': 'it',
    'portuguese': 'pt',
    'dutch': 'nl',
    'russian': 'ru',
    'arabic': 'ar',
    'chinese': 'zh',
    'japanese': 'ja',
    'korean': 'ko',
    'polish': 'pl',
}

# -------------------- Double Metaphone & word list --------------------

try:
    from metaphone import doublemetaphone
except ImportError as e:
    raise SystemExit("Missing dependency 'Metaphone'. Install with: pip install Metaphone") from e

try:
    from wordfreq import top_n_list
except ImportError as e:
    raise SystemExit("Missing dependency 'wordfreq'. Install with: pip install wordfreq") from e


# -------------------- Tokenization & casing --------------------

WORD_RE = re.compile(r"\b[\w’'-]+\b", flags=re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r'([^.!?…]*[.!?…][)"\']?\s*|[^.!?…]+$)', re.UNICODE)

def tokenize_preserve(text: str) -> List[Tuple[str, bool]]:
    """Split text into tokens preserving punctuation/spacing. Returns (token, is_word)."""
    tokens: List[Tuple[str, bool]] = []
    last = 0
    for m in WORD_RE.finditer(text):
        if m.start() > last:
            tokens.append((text[last:m.start()], False))
        tokens.append((m.group(0), True))
        last = m.end()
    if last < len(text):
        tokens.append((text[last:], False))
    return tokens

def detect_case_pattern(word: str) -> str:
    if word.islower():
        return 'lower'
    if word.isupper():
        return 'upper'
    if word[:1].isupper() and word[1:].islower():
        return 'title'
    return 'mixed'

def apply_case_pattern(sample: str, pattern: str) -> str:
    if pattern == 'lower':
        return sample.lower()
    if pattern == 'upper':
        return sample.upper()
    if pattern == 'title':
        return sample.capitalize()
    return sample  # mixed: leave as-is


# -------------------- Normalization for equality checks --------------------

_punct_re = re.compile(r"[’'`-]")

def strip_punct_and_casefold(s: str) -> str:
    """Robust normalization for equality: remove inner punctuation, strip diacritics, casefold."""
    s2 = _punct_re.sub("", s)
    s2 = unicodedata.normalize("NFKD", s2)
    s2 = "".join(ch for ch in s2 if not unicodedata.combining(ch))
    return s2.casefold()


# -------------------- Metaphone index & candidate finding --------------------

def build_metaphone_index(lang: str, max_words: int = 20_000, min_len: int = 3) -> Dict[str, List[str]]:
    """
    Build a map: Double Metaphone code -> list of words.
    Words are drawn from wordfreq's top N list for the given language.
    It also builds a fuzzy index on 3-char prefixes.
    """
    try:
        lexicon = top_n_list(lang, max_words)
    except Exception:
        # Fallback if language isn't known to wordfreq
        lexicon = top_n_list("en", max_words)

    index: Dict[str, List[str]] = {}
    seen = set()

    for w in lexicon:
        if len(w) < min_len:
            continue
        lw = w.strip()
        if not lw or lw in seen:
            continue
        seen.add(lw)
        m1, m2 = doublemetaphone(lw)
        codes = {c for c in (m1, m2) if c}
        for m in codes:
            index.setdefault(m, []).append(lw)
            # Add to fuzzy index (key starts with '_')
            prefix = m[:3]
            if len(prefix) == 3:
                index.setdefault(f"_{prefix}", []).append(lw)
    return index

def find_similar_by_metaphone(word: str, index: Dict[str, List[str]], use_fuzzy: bool = False) -> List[str]:
    """
    Combine candidates from both DM codes, remove the original (robustly).
    If use_fuzzy is True, also includes candidates from a pre-built fuzzy prefix index.
    """
    m1, m2 = doublemetaphone(word)
    codes = {c for c in (m1, m2) if c}

    pool: List[str] = []
    for code in codes:
        # Add exact matches
        pool.extend(index.get(code, []))
        # Add fuzzy matches if enabled
        if use_fuzzy:
            prefix = code[:3]
            if len(prefix) == 3:
                pool.extend(index.get(f"_{prefix}", []))

    if not pool:
        return []

    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for w in pool:
        if w not in seen:
            seen.add(w)
            deduped.append(w)

    # Filter out the original word using robust normalization
    original_key = strip_punct_and_casefold(word)
    filtered = [w for w in deduped if strip_punct_and_casefold(w) != original_key]

    return filtered


# -------------------- Text Transformation --------------------

def substitute_similar_sound(text: str, index: Dict[str, List[str]], rng: random.Random, use_fuzzy: bool = False) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Replace one random word in the text with a phonetic neighbor.
    Returns the new text, the original word, and the new replacement word.
    """
    tokens = tokenize_preserve(text)
    word_positions = [i for i, (_, is_word) in enumerate(tokens) if is_word]
    if not word_positions:
        return text, None, None

    # Find which words have candidates
    words_with_candidates = [] # List of (position, list_of_candidates)
    for pos in word_positions:
        word = tokens[pos][0]
        if sum(ch.isalpha() for ch in word) < 3:
            continue

        plain = _punct_re.sub("", word)
        candidates = find_similar_by_metaphone(plain, index, use_fuzzy=use_fuzzy)
        if candidates:
            words_with_candidates.append((pos, candidates))

    if not words_with_candidates:
        return text, None, None

    # 1. Choose which word to replace (from those with candidates)
    pos_to_replace, candidates_for_word = rng.choice(words_with_candidates)
    
    # 2. Choose which replacement to use
    replacement_word_from_lexicon = rng.choice(candidates_for_word)

    original_word = tokens[pos_to_replace][0]
    pattern = detect_case_pattern(original_word)

    final_replacement_word = apply_case_pattern(replacement_word_from_lexicon, pattern)
    tokens[pos_to_replace] = (final_replacement_word, True)

    return "".join(tok for tok, _ in tokens), original_word, final_replacement_word

def transform_text(text: str, index: Dict[str, List[str]], rng: random.Random, use_fuzzy: bool = False) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Apply one substitution to the entire text.
    """
    # Re-seed for each top-level transformation to ensure that different rows
    # get different substitutions even if the text is identical.
    seeded_rng = random.Random(rng.getrandbits(64))
    return substitute_similar_sound(text, index, rng=seeded_rng, use_fuzzy=use_fuzzy)


# -------------------- File processing (JSONL/Excel) --------------------

def process_file(
    input_path: str,
    report_col: str = "report",
    lang_col: str = "lang",
    translation_col: Optional[str] = None,
    output_path: Optional[str] = None,
    num_variants: int = 1,
    new_column_prefix: str = "_dmeta",
    in_place: bool = False,
    seed: Optional[int] = None,
    max_words_per_lang: int = 50_000,
    min_len: int = 5,
    use_fuzzy: bool = False,
    target_language: Optional[str] = None,
) -> pd.DataFrame:
    """
    For each row in the input file, generate multiple variant outputs.
    Supports .jsonl and .xlsx files.
    """
    if input_path.endswith(".jsonl"):
        df = pd.read_json(input_path, lines=True)
    elif input_path.endswith(".xlsx"):
        df = pd.read_excel(input_path)
    else:
        raise ValueError(f"Unsupported input file format: {input_path}. Please use .jsonl or .xlsx.")

    if report_col not in df.columns:
        raise ValueError(f"Column '{report_col}' not found.")
    if lang_col not in df.columns:
        raise ValueError(f"Column '{lang_col}' not found.")

    if target_language:
        df = df[df[lang_col].str.lower() == target_language.lower()].copy()
        print(f"Filtered to {len(df)} reports for language: {target_language}")

    def normalize_lang(lang_input) -> str:
        if pd.isna(lang_input):
            return "en"
        s = str(lang_input).strip().lower()
        if not s:
            return "en"

        if s in LANG_NAME_TO_CODE:
            return LANG_NAME_TO_CODE[s]

        for sep in ("-", "_"):
            if sep in s:
                s = s.split(sep, 1)[0]
        return s

    rng = random.Random(seed)
    index_cache: Dict[str, Dict[str, List[str]]] = {}

    def get_index(lang_code: str) -> Dict[str, List[str]]:
        lang_norm = normalize_lang(lang_code)
        if lang_norm not in index_cache:
            index_cache[lang_norm] = build_metaphone_index(
                lang=lang_norm, max_words=max_words_per_lang, min_len=min_len
            )
        return index_cache[lang_norm]

    eng_idx = get_index("en") if translation_col and translation_col in df.columns else None

    for variant in range(1, num_variants + 1):
        report_target_col = f"{report_col}{new_column_prefix}{variant}"
        report_out_vals = []
        report_mistakes = []
        report_now_words = []

        trans_target_col = f"{translation_col}{new_column_prefix}{variant}" if translation_col else None
        trans_out_vals = []
        trans_mistakes = []
        trans_now_words = []

        for i, row in df.iterrows():
            text = row[report_col]
            lang = row[lang_col]
            if not isinstance(text, str) or not text.strip():
                report_out_vals.append(text)
                report_mistakes.append(None)
                report_now_words.append(None)
            else:
                idx = get_index(lang)
                new_text, original_word, new_word = transform_text(text, idx, rng=rng, use_fuzzy=use_fuzzy)
                report_out_vals.append(new_text)
                report_mistakes.append(original_word)
                report_now_words.append(new_word)

            if trans_target_col and eng_idx:
                trans_text = row.get(translation_col)
                if not isinstance(trans_text, str) or not trans_text.strip():
                    trans_out_vals.append(trans_text)
                    trans_mistakes.append(None)
                    trans_now_words.append(None)
                else:
                    new_text, original_word, new_word = transform_text(trans_text, eng_idx, rng=rng, use_fuzzy=use_fuzzy)
                    trans_out_vals.append(new_text)
                    trans_mistakes.append(original_word)
                    trans_now_words.append(new_word)

        df[report_target_col] = report_out_vals
        df[f"original_mistake{variant}"] = report_mistakes
        df[f"original_now{variant}"] = report_now_words

        if trans_target_col:
            df[trans_target_col] = trans_out_vals
            df[f"english_mistake{variant}"] = trans_mistakes
            df[f"english_now{variant}"] = trans_now_words

    if output_path:
        if output_path.endswith(".jsonl"):
            df.to_json(output_path, orient='records', lines=True, force_ascii=False)
        elif output_path.endswith(".xlsx"):
            df.to_excel(output_path, index=False)
        else:
            print(f"Warning: Output path '{output_path}' has no extension. Defaulting to .jsonl format.")
            df.to_json(output_path, orient='records', lines=True, force_ascii=False)

    return df




# -------------------- CLI --------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phonetic substitution on Excel with per-row language.")
    p.add_argument("--input", type=str, required=True, help="Path to input file (.xlsx or .jsonl)")
    p.add_argument("--report-col", type=str, default="report", help="Column with text to transform.")
    p.add_argument("--lang-col", type=str, default="lang", help="Column with language codes.")
    p.add_argument("--translation-col", type=str, default=None, help="(optional) Column with English translation to also modify.")
    p.add_argument("--out", type=str, default=None, help="Optional path to save output .xlsx")
    p.add_argument("--in-place", action="store_true", help="Overwrite report column instead of creating a new one.")
    p.add_argument("--seed", type=int, default=None, help="Base RNG seed (None = new randomness).")
    p.add_argument("--max-words-per-lang", type=int, default=50_000, help="Candidate pool size per language.")
    p.add_argument("--min-len", type=int, default=3, help="Minimum word length to include in index.")
    p.add_argument("--fuzzy", action="store_true", help="Use fuzzy phonetic matching to increase candidates.")
    p.add_argument("--lang", type=str, default=None, help="Target language to process.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    df = process_file(
        input_path=args.input,
        output_path=args.out,
        report_col=args.report_col,
        lang_col=args.lang_col,
        translation_col=args.translation_col,
        num_variants=1,
        in_place=args.in_place,
        seed=args.seed,
        max_words_per_lang=args.max_words_per_lang,
        min_len=args.min_len,
        use_fuzzy=args.fuzzy,
        target_language=args.lang,
    )

    print(df.head())