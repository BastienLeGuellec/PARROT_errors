import argparse
import os
import pandas as pd
from openai import OpenAI
from llm_cache import cached_chat_completions_create
import json
import time
from typing import Optional, Tuple

# --- LLM Interaction ---

CACHE_PATH = "cache/llm_cache.sqlite"


def get_detector_prediction(client: OpenAI, text: str, model: str) -> str:
    """
    Asks the detector LLM to find an error in the text.
    Returns a JSON string with 'error_detected' (boolean) and 'explanation'.
    """
    prompt = f"""You are an expert medical proofreader. The following medical report may contain an error. Please analyze the text and provide your answer in a JSON format with two keys: 'error_detected' (a boolean value indicating if you found an error) and 'explanation' (a brief explanation of the error if found, or 'No error found.'). No need to suggest correction, just flagging the error is sufficient.

Text:
{text}"""

    try:
        # Use cache-aware wrapper which returns a parsed JSON-like dict
        resp = cached_chat_completions_create(
            client,
            CACHE_PATH,
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert medical proofreader."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        content = resp["choices"][0]["message"]["content"].strip()
        return content
    except Exception as e:
        print(f"An error occurred with the detector API: {e}")
        time.sleep(5)
        return json.dumps({"error_detected": False, "explanation": "API_ERROR"})


def get_judge_evaluation(client: OpenAI, detector_response: str, error_type: str, original_mistake: Optional[str] = None, corrected_mistake: Optional[str] = None, model: str = "gpt-5-mini") -> Tuple[int, str]:
    """
    Asks the judge LLM to evaluate the detector's response.
    Returns a tuple (score, raw_response).
    """

    if error_type == "none":
        prompt = f"""You are an expert evaluator of language models. A language model was tasked with proofreading a perfect medical report, with no error (we are sure). The model's response was: {detector_response}. Does the model's response correctly state that no error was found? Respond with only '1' if no error was found (correct response) and '0' if any error was found (incorrect response)."""
    else:
        prompt = f"""You are an expert evaluator of language models. A language model was tasked with proofreading a medical report.

The report had the following known error:
- Error Type: {error_type}
- Incorrect Word(s): {corrected_mistake}

The language model's response was:
{detector_response}

Your task is to evaluate if the model's response correctly identifies this specific error. The model should detect an error and the explanation should be consistent with the known error. I am just interested in the flagging of the mistake, not its correction.

Respond with only '1' for a correct evaluation or '0' for an incorrect one"""

    try:
        resp = cached_chat_completions_create(
            client,
            CACHE_PATH,
            model="gpt-5-mini",
            messages=[
                {"role": "system",
                    "content": "You are an expert evaluator of language models."},
                {"role": "user", "content": prompt},
            ]
        )

        raw_response = resp["choices"][0]["message"]["content"].strip()

        score = 1 if raw_response == '1' else 0
        return score, raw_response
    except Exception as e:
        print(f"An error occurred with the judge API: {e}")
        time.sleep(5)
        return 0, "API_ERROR"

# --- Main Evaluation Logic ---


def run_evaluation(input_path: str, api_key: str, output_path: str, model_name: str):
    """
    Main function to load data, query LLMs, and save evaluation results.
    """
    df = pd.read_json(input_path, lines=True)
    print(f"Loaded {len(df)} records from {input_path}")

    client = OpenAI(api_key=api_key)

    processed_languages = []
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if 'language' in record and record['language'] not in processed_languages:
                        processed_languages.append(record['language'])
                except json.JSONDecodeError:
                    continue  # ignore corrupted lines
        if processed_languages:
            print(
                f"Languages already processed: {', '.join(processed_languages)}")

    report_versions = {
        "proofread": {"error_type": "none"},
        "dmeta1": {"error_type": "phonetic/typography/wrong word"},
        "laterality": {"error_type": "laterality"},
        "negation": {"error_type": "negation"},
    }

    results_batch = []
    case_counter = 0

    for i, row in df.iterrows():
        if row['language'] in processed_languages:
            continue

        if row['language'] not in ["Italian", "French", "Greek", "German", "Polish"]:
            continue

        print(f"Processing report # {row['no']}...")
        for version, info in report_versions.items():
            # Determine which columns to use based on the version
            if version == "proofread":
                report_col = "report_proofread"
                trans_col = "translation_proofread"
            elif version == "dmeta1":
                report_col = "report_proofread_dmeta1"
                trans_col = "translation_proofread_dmeta1"
            else:
                report_col = f"report_with_{version}_error"
                trans_col = f"translation_with_{version}_error"

            texts_to_evaluate = {
                report_col: row.get(report_col),
                trans_col: row.get(trans_col)
            }

            for col, text in texts_to_evaluate.items():
                if pd.notna(text):
                    case_counter += 1
                    print(f"  Evaluating {col}...")
                    detector_response_str = get_detector_prediction(
                        client, text, model=model_name)
                    print(f"    Detector response: {detector_response_str}")
                    try:
                        detector_response_json = json.loads(
                            detector_response_str)
                    except json.JSONDecodeError:
                        detector_response_json = {
                            "error_detected": False, "explanation": "Invalid JSON response"}

                    original_mistake = None
                    corrected_mistake = None
                    if info['error_type'] != 'none':
                        if version == "dmeta1":
                            if "translation" in col:
                                original_mistake = row.get("english_mistake1")
                                corrected_mistake = row.get("english_now1")
                            else:
                                original_mistake = row.get("original_mistake1")
                                corrected_mistake = row.get("original_now1")
                        else:
                            original_mistake = row.get(
                                f"original_{version}_mistake")
                            corrected_mistake = row.get(
                                f"corrected_{version}_mistake")

                    judge_score, judge_response_raw = get_judge_evaluation(
                        client, detector_response_str, info['error_type'], original_mistake, corrected_mistake)
                    print(
                        f"    Judge score: {judge_score}, Judge response: {judge_response_raw}")

                    result = {
                        "report_no": row["no"],
                        "language": row["language"] if "translation" not in col else "english",
                        "report_version": col,
                        "error_type": info["error_type"],
                        "original_mistake": original_mistake,
                        "corrected_mistake": corrected_mistake,
                        "report_text": text,
                        "detector_response": detector_response_json,
                        "judge_score": judge_score,
                        "judge_response": judge_response_raw,
                    }
                    results_batch.append(result)

                    if case_counter % 5 == 0:
                        with open(output_path, 'a') as f:
                            for res in results_batch:
                                f.write(json.dumps(res) + '\n')
                        results_batch = []
                else:
                    print(f"  Skipping {col} (not found in data).")

    # Save any remaining results
    if results_batch:
        with open(output_path, 'a') as f:
            for res in results_batch:
                f.write(json.dumps(res) + '\n')

    print(f"Evaluation complete. Results saved to {output_path}")

# --- CLI ---


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Evaluate LLM performance on error detection.")
    parser.add_argument("--input", type=str, default="reports_with_errors.jsonl",
                        help="Path to input data file (.jsonl).")
    parser.add_argument("--output", type=str, default="evaluation_results.jsonl",
                        help="Path to base output results file (.jsonl).")
    parser.add_argument("--api-key", type=str,
                        help="OpenAI API key. Can also be set via OPENAI_API_KEY environment variable.")
    parser.add_argument("--model", type=str, default="gpt-5-mini",
                        help="The detector model to be evaluated.")

    args = parser.parse_args()
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OpenAI API key must be provided via --api-key argument or OPENAI_API_KEY environment variable.")

    # Construct the new output path with the model name suffix
    base_name, extension = os.path.splitext(args.output)
    # Sanitize model name for use in filename
    model_suffix = args.model.replace('/', '_').replace(':', '-')
    output_path_with_suffix = f"{base_name}_{model_suffix}{extension}"

    run_evaluation(
        input_path=args.input,
        api_key=api_key,
        output_path=output_path_with_suffix,
        model_name=args.model
    )
