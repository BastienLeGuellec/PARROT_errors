
import json
import os
import pandas as pd
from openai import OpenAI
import time
import argparse

# --- LLM Interaction ---

def get_proofread_text(client: OpenAI, text: str, model: str = "gpt-4o") -> str:
    """
    Asks the LLM to proofread the text and return the corrected version.
    """
    if not text or pd.isna(text):
        return ""

    prompt = f"""You are an expert medical proofreader. Your task is to proofread the following medical report text for any spelling, grammatical, or typographical errors.
- Correct any errors you find.
- Do not change the medical meaning or terminology.
- Preserve the original formatting (line breaks, spacing).
- Return only the fully corrected text (no comment)
- Original Text can be errorless: the less you change the better!

Original Text:
{text}

Corrected Text:
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert medical proofreader. Your task is to proofread the following medical report text for any spelling, grammatical, or typographical errors. Return only the fully corrected text, preserving original formatting."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=2048, # Allow for longer reports
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred with the OpenAI API: {e}")
        time.sleep(5)
        return f"API_ERROR: {e}"

# --- Main Logic ---

def proofread_dataset(input_path: str, output_path: str, api_key: str, model_name: str):
    """
    Loads the dataset, proofreads reports and translations, and saves to a new file.
    """
    try:
        df = pd.read_json(input_path, lines=True)
    except ValueError:
        print(f"Could not read {input_path} as a jsonl file. Trying to read as a single JSON array.")
        df = pd.read_json(input_path)

    print(f"Loaded {len(df)} records from {input_path}")

    client = OpenAI(api_key=api_key)

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for i, row in df.iterrows():
            if (i + 1) % 2 == 0:
                print(f"Processing row {i+1}/{len(df)}...")

            new_data_point = row.to_dict()
            
            # Define the languages to proofread
            allowed_languages = ['German', 'French', 'Italian', 'Spanish']

            if row.get("language") in allowed_languages:
                print(f"Processing row {i+1}/{len(df)} ({row.get('language')})...")
                original_report = row.get("report")
                original_translation = row.get("translation")

                # Proofread the original report
                if pd.notna(original_report):
                    print("  Proofreading original report...")
                    proofread_report = get_proofread_text(client, original_report, model=model_name)
                    new_data_point['report_proofread'] = proofread_report
                    print(original_report)
                    print(proofread_report)
                else:
                    new_data_point['report_proofread'] = ""

                # Proofread the translation
                if pd.notna(original_translation):
                    print("  Proofreading translation...")
                    proofread_translation = get_proofread_text(client, original_translation, model=model_name)
                    new_data_point['translation_proofread'] = proofread_translation
                else:
                    new_data_point['translation_proofread'] = ""
            else:
                # If the language is not in the allowed list, just keep the original data
                new_data_point['report_proofread'] = row.get("report")
                new_data_point['translation_proofread'] = row.get("translation")

            # Write the new data point to the output file
            f_out.write(json.dumps(new_data_point, ensure_ascii=False) + '\n')

    print(f"\nProofreading complete. Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proofread medical reports using an LLM.")
    parser.add_argument("--input", type=str, default="PARROT_v1_0.jsonl", help="Path to the input data file (.jsonl).")
    parser.add_argument("--output", type=str, default="parrot_proofread.jsonl", help="Path to the output data file.")
    parser.add_argument("--api-key", type=str, default="sk-sdU_3qoZa-TBKfMRIGGugsmCLkhNZFUi9QpFubFgTST3BlbkFJ2bYHLsOUeRcmSdcTOYPVHujcfuGfzRLiFqPoLNEX4A", help="OpenAI API key.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="The model to use for proofreading.")

    args = parser.parse_args()

    print(f"Starting proofreading process...")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Using model: {args.model}")

    proofread_dataset(
        input_path=args.input,
        output_path=args.output,
        api_key=args.api_key,
        model_name=args.model
    )
