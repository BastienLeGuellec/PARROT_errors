import json
import os
import pandas as pd
from openai import OpenAI
import time
import argparse

# --- LLM Interaction ---

def get_negation_error_in_pair(client: OpenAI, report_text: str, translation_text: str, model: str = "gpt-4o") -> dict:
    """
    Asks the LLM to introduce a consistent negation contradiction in a report and its translation.
    Returns a dictionary with the modified texts, or {"modified_report": "NA", "modified_translation": "NA"} if no modification was possible.
    """
    if not report_text or pd.isna(report_text) or not translation_text or pd.isna(translation_text):
        return {"modified_report": "NA", "modified_translation": "NA"}

    prompt = f"""You are an expert in medical report analysis and translation. Your task is to introduce a specific type of error - a true negation contradiction - consistently across a medical report and its translation.

A true negation contradiction occurs when a clinical finding is described as present in one section (e.g., "Findings") but absent in another (e.g., "Conclusion"), or vice-versa. I will then challenge students to spot the errors, so make them as subtle as you can. You can either change the "results section" or the "impression section".

Your instructions are very strict:
1.  First, analyze the Original Report to identify if it *already contains* at least two distinct sections: a main descriptive section (e.g., "Findings", "Description") and a final summary section (e.g., "Conclusion", "Impression", "Summary").
2.  **YOU ARE NOT ALLOWED TO ADD A NEW SUMMARY OR CONCLUSION SECTION IF IT DOES NOT ALREADY EXIST.** If the original report does NOT already contain both a descriptive and a summary section, you cannot create a contradiction. You must immediately stop and return a JSON object where the value for both "modified_report" and "modified_translation" is "NA".
3.  If the structure exists, search for a clinical finding that is mentioned in BOTH the descriptive section AND the summary section.
4.  If no such recurring finding exists, you cannot create a contradiction. You must stop and return the "NA" JSON object as described above.
5.  If you find a recurring finding, you must then introduce a subtle negation contradiction by performing one of the following changes. Please focus on sentences that keep their meaning after the change. For example, changing "Left sylvian ischemic stroke" to "No left sylvian ischemic stroke" is dumb.
    a.  **Introduce a Negation:** In the Original Report, add a negation to the finding either in the results or summary sections (e.g., change "evidence of pneumonia" to "no evidence of pneumonia"). 
    b.  **Remove a Negation:** If the finding is already negated in both sections (e.g., "no effusion"), remove the negation either in the summary or results sections (e.g., change "no effusion" to "small effusion"). 
6.  In the Original Translation, apply the EXACT SAME conceptual change. 
7.  **IMPORTANT:** The modification must be as minimal as possible, ideally changing only a single word (e.g., 'with' to 'without', 'clear' to 'unclear', adding/removing 'no' or 'not'). The goal is to create a subtle error that could plausibly be a speech recognition mistake. Avoid rewriting entire phrases.
8.  You must return a JSON object with two keys: "modified_report" and "modified_translation".
9.  If you successfully introduced the error as described, the values should be the full, modified texts.
10. If you could not introduce an error for any reason (missing sections, no recurring finding, etc.), the value for both keys must be the exact string "NA".
11. Do not add any comments or explanations outside of the JSON structure.

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
                {"role": "system", "content": "You are an expert in medical report analysis. Your task is to introduce a consistent negation contradiction into a report and its translation, returning a JSON object with the results."},
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
    Loads dataset, introduces consistent negation errors in reports and their translations,
    and saves to a new file. Resumes from the highest 'no' + 1 in the output file.
    """
    try:
        df = pd.read_json(input_path, lines=True)
        if 'no' in df.columns:
            df['no'] = pd.to_numeric(df['no'])
    except ValueError:
        print(f"Could not read {input_path} as a jsonl file. Trying to read as a single JSON array.")
        df = pd.read_json(input_path)
        if 'no' in df.columns:
            df['no'] = pd.to_numeric(df['no'])

    print(f"Loaded {len(df)} total records from {input_path}")

    start_from_no = 0
    if os.path.exists(output_path):
        max_no = -1
        try:
            with open(output_path, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    if line.strip():
                        record = json.loads(line)
                        if 'no' in record:
                            current_no = int(record['no'])
                            if current_no > max_no:
                                max_no = current_no
            if max_no > -1:
                start_from_no = max_no + 1
                print(f"Output file found. Highest 'no' is {max_no}. Resuming from 'no' {start_from_no}.")
            else:
                print("Output file is empty. Starting from the beginning.")
        except Exception as e:
            print(f"Warning: Could not process existing output file '{output_path}'. Starting from scratch. Error: {e}")
            start_from_no = 0
    else:
        print("No existing output file found. Starting from the beginning.")

    client = OpenAI(api_key=api_key)

    newly_modified_count = 0
    allowed_languages = ['French', 'German', 'Spanish', 'Italian']
    print(f"Processing only languages: {allowed_languages}")

    with open(output_path, 'a', encoding='utf-8') as f_out:
        # Sort dataframe by 'no' to ensure sequential processing
        for i, row in df.sort_values(by='no').iterrows():
            record_no = row.get("no")
            if record_no < start_from_no:
                continue

            print(f"Processing record with no='{record_no}'...")

            language = row.get("language")
            if language not in allowed_languages:
                print(f"  -> Skipping (language: {language or 'N/A'} is not in the allowed list).")
                continue

            original_report = row.get("report_proofread")
            original_translation = row.get("translation_proofread")

            if pd.notna(original_report) and original_report.strip() and pd.notna(original_translation) and original_translation.strip():
                result = get_negation_error_in_pair(client, original_report, original_translation, model=model_name)
                modified_report = result.get("modified_report")
                modified_translation = result.get("modified_translation")

                if modified_report != "NA" and modified_translation != "NA":
                    newly_modified_count += 1
                    print(f"  -> Negation error introduced for no='{record_no}' ({language}).")

                    print("\n--- ORIGINAL REPORT ---")
                    print(original_report)
                    print("\n--- MODIFIED REPORT ---")
                    print(modified_report)
                    print("\n--- ORIGINAL TRANSLATION ---")
                    print(original_translation)
                    print("\n--- MODIFIED TRANSLATION ---")
                    print(modified_translation)
                    print("\n" + "="*40 + "\n")
                    
                    output_record = row.to_dict()
                    output_record['report_with_error'] = modified_report
                    output_record['translation_with_error'] = modified_translation
                    
                    f_out.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                else:
                    print(f"  -> No consistent error could be introduced for no='{record_no}' ({language}).")
            else:
                print(f"  -> Skipping no='{record_no}' ({language}) due to missing report or translation.")

            if (newly_modified_count + 1) % 2 == 0:
                print(f"  ...Saving progress to disk...")
                f_out.flush()

    print(f"\nProcessing complete.")
    print(f"Introduced errors in {newly_modified_count} new records during this run.")
    
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f_in:
            total_in_output = sum(1 for _ in f_in)
        print(f"Total modified records in {output_path}: {total_in_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Introduce negation errors into medical reports using an LLM.")
    parser.add_argument("--input", type=str, default="parrot_proofread.jsonl", help="Path to the input data file (.jsonl).")
    parser.add_argument("--output", type=str, default="reports_with_negation_errors.jsonl", help="Path to the output data file for modified reports.")
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
