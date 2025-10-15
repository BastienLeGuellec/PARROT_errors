import json
import os
import pandas as pd
from openai import OpenAI
import time
import argparse

# --- LLM Interaction ---

def get_laterality_error_in_pair(client: OpenAI, report_text: str, translation_text: str, model: str = "gpt-4o") -> dict:
    """
    Asks the LLM to introduce a consistent, true laterality contradiction in a report and its translation.
    Returns a dictionary with the modified texts, or {"modified_report": "NA", "modified_translation": "NA"} if no modification was possible.
    """
    if not report_text or pd.isna(report_text) or not translation_text or pd.isna(translation_text):
        return {"modified_report": "NA", "modified_translation": "NA"}

    prompt = f"""You are an expert in medical report analysis and translation. Your task is to introduce a specific type of error - a true laterality contradiction - consistently across a medical report and its translation. A simple variation is not acceptable.

A true laterality contradiction occurs when a finding is described on one side in the main descriptive section (e.g., "Findings") but is described on the opposite side in the final summary section (e.g., "Conclusion" or "Impression").

Your instructions are very strict:
1.  First, analyze the Original Report to identify if it *already contains* at least two distinct sections: a main descriptive section (e.g., "Findings", "Description") and a final summary section (e.g., "Conclusion", "Impression", "Summary").
2.  **YOU ARE NOT ALLOWED TO ADD A NEW SUMMARY OR CONCLUSION SECTION IF IT DOES NOT ALREADY EXIST.** If the original report does NOT already contain both a descriptive and a summary section, you cannot create a contradiction. You must immediately stop and return a JSON object where the value for both "modified_report" and "modified_translation" is "NA".
3.  If the structure exists, search for a unilateral finding (a finding on the "left" or "right") that is mentioned in BOTH the descriptive section AND the summary section.
4.  If no such recurring unilateral finding exists, you cannot create a contradiction. You must stop and return the "NA" JSON object as described above.
5.  If you find a recurring unilateral finding, you must then perform the following change:
    a.  In the Original Report, change the laterality (left->right or right->left) ONLY in the summary section (e.g., "Conclusion"). LEAVE THE DESCRIPTIVE SECTION UNCHANGED.
    b.  In the Original Translation, apply the EXACT SAME conceptual change. Modify the laterality ONLY in the corresponding summary section of the translation. LEAVE THE DESCRIPTIVE SECTION OF THE TRANSLATION UNCHANGED.
6.  You must return a JSON object with two keys: "modified_report" and "modified_translation".
7.  If you successfully introduced the error as described, the values should be the full, modified texts.
8.  If you could not introduce an error for any reason (missing sections, no recurring finding, etc.), the value for both keys must be the exact string "NA".
9.  Do not add any comments or explanations outside of the JSON structure.

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
                {"role": "system", "content": "You are an expert in medical report analysis. Your task is to introduce a consistent laterality contradiction into a report and its translation, returning a JSON object with the results."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"}, # Use JSON mode
        )
        
        result_json = response.choices[0].message.content
        result_data = json.loads(result_json)
        
        # Basic validation
        if 'modified_report' in result_data and 'modified_translation' in result_data:
            return result_data
        else:
            # Malformed JSON from LLM
            print("  -> Warning: LLM returned malformed JSON. Skipping.")
            return {"modified_report": "NA", "modified_translation": "NA"}

    except Exception as e:
        print(f"An error occurred with the OpenAI API or JSON parsing: {e}")
        time.sleep(5)
        # Return "NA" on error to signal failure
        return {"modified_report": "NA", "modified_translation": "NA"}

# --- Main Logic ---

def introduce_errors_in_dataset(input_path: str, output_path: str, api_key: str, model_name: str):
    """
    Loads dataset, introduces consistent laterality errors in reports and their translations,
    and saves to a new file. Only items where an error could be introduced in both are saved.
    """
    try:
        df = pd.read_json(input_path, lines=True)
    except ValueError:
        print(f"Could not read {input_path} as a jsonl file. Trying to read as a single JSON array.")
        df = pd.read_json(input_path)

    print(f"Loaded {len(df)} records from {input_path}")

    client = OpenAI(api_key=api_key)

    modified_count = 0
    processed_languages = []
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if 'language' in record and record['language'] not in processed_languages:
                            processed_languages.append(record['language'])
                    except json.JSONDecodeError:
                        continue
        if processed_languages:
            print(f"Languages already processed: {', '.join(processed_languages)}")

    allowed_languages = ['French', 'German', 'Spanish', 'Italian', 'Polish']
    print(f"Processing only languages: {', '.join(allowed_languages)}")

    with open(output_path, 'a', encoding='utf-8') as f_out:
        for i, row in df.iterrows():
            print(f"Processing row {i+1}/{len(df)}...")

            language = row.get("language")
            if language in processed_languages:
                print(f"  -> Skipping row {i+1} (language: {language}) as it is already processed.")
                continue

            if language not in allowed_languages:
                print(f"  -> Skipping row {i+1} (language: {language or 'N/A'}) is not in the allowed list.")
                continue

            original_report = row.get("proofread_report")
            original_translation = row.get("proofread_translation")

            # Check if we have both report and translation to work with
            if pd.notna(original_report) and original_report.strip() and pd.notna(original_translation) and original_translation.strip():
                
                result = get_laterality_error_in_pair(client, original_report, original_translation, model=model_name)
                
                modified_report = result.get("modified_report")
                modified_translation = result.get("modified_translation")

                # Check if the LLM returned valid, modified reports (not "NA")
                if modified_report != "NA" and modified_translation != "NA":
                    modified_count += 1
                    print(f"  -> Laterality error introduced in report and translation {i+1} ({language}).")
                    
                    output_record = row.to_dict()
                    output_record['report_with_error'] = modified_report
                    output_record['translation_with_error'] = modified_translation
                    
                    f_out.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                else:
                    print(f"  -> No consistent error could be introduced for row {i+1} ({language}).")
            else:
                print(f"  -> Skipping row {i+1} ({language}) due to missing report or translation.")

            # After every 2 processed reports, flush the buffer to disk to save progress.
            if (i + 1) % 2 == 0:
                print(f"  ...Saving progress to disk...")
                f_out.flush()

    print(f"\nProcessing complete.")
    print(f"Introduced errors in {modified_count} out of {len(df)} processed report/translation pairs.")
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Introduce laterality errors into medical reports using an LLM.")
    parser.add_argument("--input", type=str, default="parrot_proofread.jsonl", help="Path to the input data file (.jsonl).")
    parser.add_argument("--output", type=str, default="reports_with_laterality_errors.jsonl", help="Path to the output data file for modified reports.")
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API key. Defaults to OPENAI_API_KEY environment variable.")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="The model to use for introducing errors.")

    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or provide it using the --api-key argument.")

    print(f"Starting error introduction process...")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Using model: {args.model}")

    introduce_errors_in_dataset(
        input_path=args.input,
        output_path=args.output,
        api_key=args.api_key,
        model_name=args.model
    )
