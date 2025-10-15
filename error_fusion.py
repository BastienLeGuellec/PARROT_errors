#!/usr/bin/env python3
"""
This script fuses three error introduction mechanisms into a single pipeline:
1. Phonetic errors using Double Metaphone.
2. Laterality contradiction errors using an LLM.
3. Negation contradiction errors using an LLM.

It processes a large JSONL file line-by-line, attempts to introduce all three
error types for each report and its English translation, and saves the results
to a new JSONL file.

This version supports resuming from a partially completed output file.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import time
import unicodedata
from typing import Dict, List, Tuple, Optional

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# --- Dependency Check ---
try:
    from metaphone import doublemetaphone
except ImportError as e:
    raise SystemExit("Missing dependency 'Metaphone'. Install with: pip install Metaphone") from e

try:
    from wordfreq import top_n_list
except ImportError as e:
    raise SystemExit("Missing dependency 'wordfreq'. Install with: pip install wordfreq") from e

# ==============================================================================
# PART 1: PHONETIC ERROR INTRODUCTION (from script.py)
# ==============================================================================

LANG_NAME_TO_CODE = {
    'english': 'en', 'french': 'fr', 'spanish': 'es', 'german': 'de',
    'italian': 'it', 'portuguese': 'pt', 'dutch': 'nl', 'russian': 'ru',
    'arabic': 'ar', 'chinese': 'zh', 'japanese': 'ja', 'korean': 'ko',
    'polish': 'pl', 'greek': 'el',
}

WORD_RE = re.compile(r"\b[\w’'-]+\b", flags=re.UNICODE)
_punct_re = re.compile(r"[’'`-]")

def normalize_lang(lang_input: str) -> str:
    """Normalizes a language name or code to a two-letter code for wordfreq."""
    if pd.isna(lang_input): return "en"
    s = str(lang_input).strip().lower()
    if not s: return "en"
    if s in LANG_NAME_TO_CODE: return LANG_NAME_TO_CODE[s]
    for sep in ("-", "_"):
        if sep in s: s = s.split(sep, 1)[0]
    return s

def tokenize_preserve(text: str) -> List[Tuple[str, bool]]:
    """Split text into tokens preserving punctuation/spacing. Returns (token, is_word)."""
    tokens: List[Tuple[str, bool]] = []
    last = 0
    for m in WORD_RE.finditer(text):
        if m.start() > last: tokens.append((text[last:m.start()], False))
        tokens.append((m.group(0), True))
        last = m.end()
    if last < len(text): tokens.append((text[last:], False))
    return tokens

def detect_case_pattern(word: str) -> str:
    if word.islower(): return 'lower'
    if word.isupper(): return 'upper'
    if word[:1].isupper() and word[1:].islower(): return 'title'
    return 'mixed'

def apply_case_pattern(sample: str, pattern: str) -> str:
    if pattern == 'lower': return sample.lower()
    if pattern == 'upper': return sample.upper()
    if pattern == 'title': return sample.capitalize()
    return sample

def strip_punct_and_casefold(s: str) -> str:
    """Robust normalization for equality checks."""
    s2 = _punct_re.sub("", s)
    s2 = unicodedata.normalize("NFKD", s2)
    s2 = "".join(ch for ch in s2 if not unicodedata.combining(ch))
    return s2.casefold()

def build_metaphone_index(lang: str, max_words: int = 50_000, min_len: int = 3) -> Dict[str, List[str]]:
    """Builds a map from a Double Metaphone code to a list of words."""
    try:
        lexicon = top_n_list(lang, max_words)
    except Exception:
        lexicon = top_n_list("en", max_words)

    index: Dict[str, List[str]] = {}
    seen = set()
    for w in lexicon:
        if len(w) < min_len: continue
        lw = w.strip()
        if not lw or lw in seen: continue
        seen.add(lw)
        m1, m2 = doublemetaphone(lw)
        codes = {c for c in (m1, m2) if c}
        for m in codes:
            index.setdefault(m, []).append(lw)
    return index

def find_similar_by_metaphone(word: str, index: Dict[str, List[str]]) -> List[str]:
    """Finds phonetic candidates for a word."""
    m1, m2 = doublemetaphone(word)
    codes = {c for c in (m1, m2) if c}
    pool: List[str] = []
    for code in codes:
        pool.extend(index.get(code, []))
    if not pool: return []
    seen = set()
    deduped = [w for w in pool if not (w in seen or seen.add(w))]
    original_key = strip_punct_and_casefold(word)
    return [w for w in deduped if strip_punct_and_casefold(w) != original_key]

def transform_text_phonetic(text: str, index: Dict[str, List[str]], rng: random.Random) -> Tuple[str, Optional[str], Optional[str]]:
    """Replace one random word in the text with a phonetic neighbor."""
    if not isinstance(text, str) or not text.strip():
        return text, None, None
    tokens = tokenize_preserve(text)
    word_positions = [i for i, (_, is_word) in enumerate(tokens) if is_word]
    if not word_positions: return text, None, None

    words_with_candidates = []
    for pos in word_positions:
        word = tokens[pos][0]
        if sum(ch.isalpha() for ch in word) < 3: continue
        plain = _punct_re.sub("", word)
        candidates = find_similar_by_metaphone(plain, index)
        if candidates: words_with_candidates.append((pos, candidates))

    if not words_with_candidates: return text, None, None

    pos_to_replace, candidates_for_word = rng.choice(words_with_candidates)
    replacement_word = rng.choice(candidates_for_word)
    original_word = tokens[pos_to_replace][0]
    pattern = detect_case_pattern(original_word)
    final_replacement = apply_case_pattern(replacement_word, pattern)
    tokens[pos_to_replace] = (final_replacement, True)
    return "".join(tok for tok, _ in tokens), original_word, final_replacement

# ==============================================================================
# PART 2: LLM-BASED ERROR INTRODUCTION
# ==============================================================================

def get_laterality_error_in_pair(client: OpenAI, report_text: str, translation_text: str, model: str) -> dict:
    """Asks the LLM to introduce a consistent, true laterality contradiction."""
    if not all(isinstance(t, str) and t.strip() for t in [report_text, translation_text]):
        return {"modified_report": "NA", "modified_translation": "NA"}
    
    prompt = f"""Your task is to introduce a laterality contradiction into a medical report. This is a test of your ability to follow instructions precisely.

**Step 1: Structural Analysis.**
First, and most importantly, examine the "Original Report". Does it contain BOTH a descriptive section (like "Findings", "Description") AND a separate, distinct summary section at the end (like "Conclusion", "Impression", "Summary")?

**Step 2: Decision.**
- If the answer to Step 1 is NO (the report lacks a separate summary section), then you are forbidden from making any changes. Your task is to immediately stop and return the JSON object: {{"modified_report": "NA", "modified_translation": "NA"}}. Do not proceed to Step 3.
- If the answer to Step 1 is YES, you may proceed to Step 3.

**Step 3: Introduce Error.**
If and only if you have proceeded past Step 2, find a unilateral finding (a "left" or "right" finding) that is mentioned in BOTH the descriptive section AND the summary section. If no such finding exists, return the "NA" JSON object. Otherwise, change the laterality (left->right or right->left) ONLY in the summary section of the Original Report and apply the exact same conceptual change to the summary of the Original Translation.

Return a JSON object with the full modified texts. Do not add any other commentary.

Original Report:
---
{report_text}
---
Original Translation:
---
{translation_text}
---
JSON Output:
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert medical editor. Your task is to follow strict instructions to either modify a report or determine it cannot be modified."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        if 'modified_report' in result and 'modified_translation' in result:
            return result
        return {"modified_report": "NA", "modified_translation": "NA"}
    except Exception as e:
        tqdm.write(f"  -> LLM call for laterality failed: {e}")
        time.sleep(5)
        return {"modified_report": "NA", "modified_translation": "NA"}

def get_negation_error_in_pair(client: OpenAI, report_text: str, translation_text: str, model: str) -> dict:
    """Asks the LLM to introduce a consistent, subtle negation contradiction."""
    if not all(isinstance(t, str) and t.strip() for t in [report_text, translation_text]):
        return {"modified_report": "NA", "modified_translation": "NA"}

    prompt = f"""Your task is to introduce a negation contradiction into a medical report. This is a test of your ability to follow instructions precisely.

**Step 1: Structural Analysis.**
First, and most importantly, examine the "Original Report". Does it contain BOTH a descriptive section (like "Findings", "Description") AND a separate, distinct summary section at the end (like "Conclusion", "Impression", "Summary")?

**Step 2: Decision.**
- If the answer to Step 1 is NO (the report lacks a separate summary section), then you are forbidden from making any changes. Your task is to immediately stop and return the JSON object: {{"modified_report": "NA", "modified_translation": "NA"}}. Do not proceed to Step 3.
- If the answer to Step 1 is YES, you may proceed to Step 3.

**Step 3: Introduce Error.**
If and only if you have proceeded past Step 2, find a clinical finding that is mentioned in BOTH the descriptive section AND the summary section. If no such finding exists, return the "NA" JSON object. Otherwise, introduce a subtle negation contradiction by adding or removing a negation in ONE of the sections. The change must be minimal (e.g., change 'with' to 'without'). Apply the exact same conceptual change to the Original Translation.

Return a JSON object with the full modified texts. Do not add any other commentary.

Original Report:
---
{report_text}
---
Original Translation:
---
{translation_text}
---
JSON Output:
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert medical editor. Your task is to follow strict instructions to either modify a report or determine it cannot be modified."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        if 'modified_report' in result and 'modified_translation' in result:
            return result
        return {"modified_report": "NA", "modified_translation": "NA"}
    except Exception as e:
        tqdm.write(f"  -> LLM call for negation failed: {e}")
        time.sleep(5)
        return {"modified_report": "NA", "modified_translation": "NA"}

# ==============================================================================
# PART 3: MAIN PROCESSING LOGIC
# ==============================================================================

def process_reports(input_path: str, output_path: str, api_key: str, model: str, seed: Optional[int]):
    """
    Main function to process reports from the input file and generate errors.
    This version supports resuming from a partially completed output file.
    """
    print("Starting report processing...")
    client = OpenAI(api_key=api_key)
    rng = random.Random(seed)
    metaphone_indices: Dict[str, Dict[str, List[str]]] = {}
    allowed_languages = ['German', 'French', 'Italian', 'Greek', 'Polish']
    print(f"Processing only the following languages: {', '.join(allowed_languages)}")

    processed_ids = set()
    if os.path.exists(output_path):
        print(f"Resuming from existing output file: {output_path}")
        try:
            with open(output_path, 'r', encoding='utf-8') as f_handle:
                for line in f_handle:
                    try:
                        record = json.loads(line)
                        if 'no' in record:
                            processed_ids.add(record['no'])
                    except json.JSONDecodeError:
                        continue
            print(f"Found {len(processed_ids)} already processed reports. They will be skipped.")
        except FileNotFoundError:
            pass # It's okay if the file doesn't exist yet

    try:
        with open(input_path, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
            
            lines_to_process = []
            for line in lines:
                try:
                    record = json.loads(line)
                    if record.get('language') in allowed_languages:
                        if record.get('no') not in processed_ids:
                            lines_to_process.append(line)
                except json.JSONDecodeError:
                    tqdm.write(f"Skipping malformed JSON line: {line.strip()}")

            print(f"Found {len(lines_to_process)} reports to process.")

            with open(output_path, 'a', encoding='utf-8') as f_out:
                for line in tqdm(lines_to_process, desc="Processing reports"):
                    record = json.loads(line)

                    output_record = record.copy()
                    report_no = record.get('no')
                    original_report = record.get("report_proofread")
                    original_translation = record.get("translation_proofread")
                    lang = record.get("language")

                    tqdm.write(f"\n--- Processing Report ID: {report_no} (Language: {lang}) ---")

                    # --- 1. Phonetic Error ---
                    if output_record.get('report_with_phonetic_error'):
                        tqdm.write("  - Phonetic error: Already exists. Skipping.")
                    else:
                        tqdm.write("  - Phonetic error: Generating...")
                        lang_code = normalize_lang(lang)
                        if lang_code not in metaphone_indices:
                            metaphone_indices[lang_code] = build_metaphone_index(lang_code)
                        
                        rep_phon, rep_orig, rep_new = transform_text_phonetic(original_report, metaphone_indices[lang_code], rng)
                        output_record['report_with_phonetic_error'] = rep_phon
                        output_record['phonetic_error_original_word'] = rep_orig
                        output_record['phonetic_error_replacement_word'] = rep_new
                        
                        if 'en' not in metaphone_indices:
                            metaphone_indices['en'] = build_metaphone_index('en')
                        
                        tra_phon, tra_orig, tra_new = transform_text_phonetic(original_translation, metaphone_indices['en'], rng)
                        output_record['translation_with_phonetic_error'] = tra_phon
                        output_record['phonetic_error_original_word_en'] = tra_orig
                        output_record['phonetic_error_replacement_word_en'] = tra_new

                    # --- 2. Laterality Error ---
                    if output_record.get('report_with_laterality_error'):
                        tqdm.write(f"  - Laterality error: Already exists. Skipping.")
                    else:
                        tqdm.write(f"  - Laterality error: Attempting...")
                        lat_result = get_laterality_error_in_pair(client, original_report, original_translation, model)
                        if lat_result.get("modified_report") != "NA":
                            tqdm.write(f"    -> Success: Laterality error introduced.")
                            tqdm.write("    --- ORIGINAL TRANSLATION ---")
                            tqdm.write(f"    {original_translation}")
                            tqdm.write("    --- MODIFIED TRANSLATION (LATERALITY) ---")
                            tqdm.write(f"    {lat_result.get('modified_translation')}")
                            output_record['report_with_laterality_error'] = lat_result.get("modified_report")
                            output_record['translation_with_laterality_error'] = lat_result.get("modified_translation")
                        else:
                            tqdm.write(f"    -> Failed: Could not introduce laterality error.")
                            output_record['report_with_laterality_error'] = None
                            output_record['translation_with_laterality_error'] = None

                    # --- 3. Negation Error ---
                    if output_record.get('report_with_negation_error'):
                        tqdm.write(f"  - Negation error: Already exists. Skipping.")
                    else:
                        tqdm.write(f"  - Negation error: Attempting...")
                        neg_result = get_negation_error_in_pair(client, original_report, original_translation, model)
                        if neg_result.get("modified_report") != "NA":
                            tqdm.write(f"    -> Success: Negation error introduced.")
                            tqdm.write("    --- ORIGINAL TRANSLATION ---")
                            tqdm.write(f"    {original_translation}")
                            tqdm.write("    --- MODIFIED TRANSLATION (NEGATION) ---")
                            tqdm.write(f"    {neg_result.get('modified_translation')}")
                            output_record['report_with_negation_error'] = neg_result.get("modified_report")
                            output_record['translation_with_negation_error'] = neg_result.get("modified_translation")
                        else:
                            tqdm.write(f"    -> Failed: Could not introduce negation error.")
                            output_record['report_with_negation_error'] = None
                            output_record['translation_with_negation_error'] = None

                    f_out.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                    f_out.flush()

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return

    print("\nProcessing complete.")
    print(f"Output with newly generated errors is saved to {output_path}")

# ==============================================================================
# PART 4: COMMAND-LINE INTERFACE
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Introduce phonetic, laterality, and negation errors into medical reports.")
    parser.add_argument("--input", type=str, default="merged_reports.jsonl", help="Path to the input data file (e.g., merged_reports.jsonl).")
    parser.add_argument("--output", type=str, default="reports_with_errors.jsonl", help="Path to the output data file.")
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API key. Defaults to OPENAI_API_KEY environment variable.")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="The model to use for introducing errors.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for phonetic error generation.")

    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or use the --api-key argument.")

    process_reports(
        input_path=args.input,
        output_path=args.output,
        api_key=args.api_key,
        model=args.model,
        seed=args.seed
    )