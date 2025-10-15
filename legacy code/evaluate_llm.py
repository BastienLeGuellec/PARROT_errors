#!/usr/bin/env python3
"""
Evaluates an LLM's ability to detect single phonetic errors in reports.

This script takes the output file from `script.py`, which contains original
reports, their English translations, and phonetically modified versions.

It iterates through each text, asks a specified LLM (e.g., GPT-4o) to find
the single incorrect word, and compares the LLM's prediction to the ground
truth.

Finally, it calculates and plots performance metrics:
- Sensitivity (Recall)
- Specificity
- F1 Score

Dependencies:
    pip install pandas openai matplotlib scikit-learn

CLI Example:
    python evaluate_llm.py \
        --input reports_dmeta.jsonl \
        --api-key "YOUR_OPENAI_API_KEY" \
        --num-variants 10
"""
import argparse
import os
import pandas as pd
from openai import OpenAI
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time

# --- LLM Interaction ---

def get_llm_prediction(client: OpenAI, text: str, model: str = "gpt-4o-mini") -> str:
    """
    Asks the LLM to find a single incorrect word in the text.

    Returns:
        The incorrect word or "N/A" if no error is found.
    """
    prompt = f"""You are an expert medical proofreader. The following medical report text may contain an error. Your task is to identify if an error exists.
- If you find an incorrect word, return only the incorrect word, with no comment.
- If the text appears to be entirely correct and contains no such errors, you must return the exact string 'N/A'.

Text:
\"{text}\""""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert medical proofreader."}, 
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=20,
        )
        prediction = response.choices[0].message.content.strip()
        print(prediction)
        # Normalize prediction for easier comparison
        if prediction != "N/A":
            prediction = prediction.strip('.,;:\"\'").lower()')
        return prediction
    except Exception as e:
        print(f"An error occurred with the OpenAI API: {e}")
        # Wait for a moment before retrying or skipping
        time.sleep(5)
        return "API_ERROR"

def normalize_for_comparison(word: str) -> str:
    """Lowercase and strip punctuation for comparing LLM output to ground truth."""
    if not isinstance(word, str):
        return ""
    return word.strip('.,;:\"\'").lower()')

# --- Main Evaluation Logic ---

def evaluate_performance(input_path: str, api_key: str, num_variants: int):
    """
    Main function to load data, query LLM, and evaluate performance.
    """
    # 1. Load Data
    if input_path.endswith(".jsonl"):
        df = pd.read_json(input_path, lines=True)
    elif input_path.endswith(".xlsx"):
        df = pd.read_excel(input_path)
    else:
        raise ValueError("Unsupported input file. Use .jsonl or .xlsx")

    print(f"Loaded {len(df)} records from {input_path}")

    # Filter for specified languages
    languages = ['Spanish', 'Italian', 'French', 'German']
    df = df[df['language'].isin(languages)]
    print(f"Filtered to {len(df)} records for languages: {', '.join(languages)}")

    # 2. Initialize
    client = OpenAI(api_key=api_key)
    all_languages = languages + ['english']
    results = {lang: {'y_true': [], 'y_pred': []} for lang in all_languages}
    results['overall'] = {'y_true': [], 'y_pred': []}

    # 3. Loop through data and evaluate
    for i, row in df.iterrows():
        print(f"Processing row {i+1}/{len(df)}...")
        lang = row['language']

        # --- Evaluate Negative Cases (Originals) ---
        # These texts should have no errors.
        original_texts = {
            "report": (row.get("report"), lang),
            "translation": (row.get("translation"), 'english')
        }
        for col_name, (text, current_lang) in original_texts.items():
            if pd.notna(text):
                if current_lang not in results:
                    results[current_lang] = {'y_true': [], 'y_pred': []}
                results[current_lang]['y_true'].append(0) # Ground truth: No error
                results['overall']['y_true'].append(0)
                prediction = get_llm_prediction(client, text)
                pred_result = 0 if prediction == "N/A" else 1
                results[current_lang]['y_pred'].append(pred_result)
                results['overall']['y_pred'].append(pred_result)

        # --- Evaluate Positive Cases (Modified Variants) ---
        # These texts are expected to have one error.
        for variant in range(1, num_variants + 1):
            # Check original report's variant
            modified_report_col = f"report_proofread_dmeta{variant}"
            mistake_report_col = f"original_mistake{variant}"
            mistake_report_col2 = f"original_now{variant}"
            if modified_report_col in df.columns and pd.notna(row.get(modified_report_col)):
                results[lang]['y_true'].append(1) # Ground truth: Error exists
                results['overall']['y_true'].append(1)
                text = row[modified_report_col]
                ground_truth = normalize_for_comparison(row.get(mistake_report_col))+normalize_for_comparison(row.get(mistake_report_col2))
                prediction = get_llm_prediction(client, text)
                # Prediction is correct if it matches the ground truth word
                pred_result = 1 if prediction in ground_truth else 0
                results[lang]['y_pred'].append(pred_result)
                results['overall']['y_pred'].append(pred_result)
                

            # Check translation's variant
            modified_trans_col = f"translation_proofread_dmeta{variant}"
            mistake_trans_col = f"english_mistake{variant}"
            mistake_trans_col2 = f"english_now{variant}"
            if modified_trans_col in df.columns and pd.notna(row.get(modified_trans_col)):
                results['english']['y_true'].append(1) # Ground truth: Error exists
                results['overall']['y_true'].append(1)
                text = row[modified_trans_col]
                ground_truth = normalize_for_comparison(row.get(mistake_trans_col))+normalize_for_comparison(row.get(mistake_trans_col2))
                prediction = get_llm_prediction(client, text)
                pred_result = 1 if prediction in ground_truth else 0
                results['english']['y_pred'].append(pred_result)
                results['overall']['y_pred'].append(pred_result)

    # 4. Calculate and Display Metrics
    all_metrics = {}
    for lang in all_languages + ['overall']:
        data = results.get(lang)
        if not data:
            continue
        y_true = data['y_true']
        y_pred = data['y_pred']

        if not y_true:
            print(f"\nNo data to evaluate for {lang.capitalize()}. Skipping.")
            continue

        # Ensure y_pred has the same length as y_true, fill with 0 if needed
        if len(y_pred) < len(y_true):
            y_pred.extend([0] * (len(y_true) - len(y_pred)))
            
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        all_metrics[lang] = {
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Precision': precision,
            'F1 Score': f1_score,
            'True Positives': tp,
            'False Positives': fp,
            'True Negatives': tn,
            'False Negatives': fn
        }

        print(f"\n--- LLM Performance Metrics ({lang.capitalize()}) ---")
        print(f"Sensitivity (Recall): {sensitivity:.2f}")
        print(f"Specificity:          {specificity:.2f}")
        print(f"Precision:            {precision:.2f}")
        print(f"F1 Score:             {f1_score:.2f}")
        print("-----------------------------")
        print(f"True Positives:  {tp}")
        print(f"False Positives: {fp}")
        print(f"True Negatives:  {tn}")
        print(f"False Negatives: {fn}")
        print("-----------------------------")

    # 5. Plot Overall Results
    if 'overall' in all_metrics:
        overall_metrics = all_metrics['overall']
        metrics_data = {
            'Sensitivity': overall_metrics['Sensitivity'],
            'Specificity': overall_metrics['Specificity'],
            'F1 Score': overall_metrics['F1 Score']
        }
        names = list(metrics_data.keys())
        values = list(metrics_data.values())

        plt.figure(figsize=(8, 6))
        bars = plt.bar(names, values, color=['#4A90E2', '#50E3C2', '#F5A623'])
        plt.ylabel('Score')
        plt.title('Overall LLM Performance in Phonetic Error Detection')
        plt.ylim(0, 1.1)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('llm_performance.png')
        print("\nOverall performance graph saved to llm_performance.png")


# --- CLI ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on error detection.")
    parser.add_argument("--input", type=str, required=True, help="Path to input data file (.jsonl or .xlsx).")
    parser.add_argument("--api-key", type=str, help="OpenAI API key. Can also be set via OPENAI_API_KEY environment variable.")
    parser.add_argument("--num-variants", type=int, default=10, help="Number of modified variants to check per report.")
    
    args = parser.parse_args()
    api_key=os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OpenAI API key must be provided via --api-key argument or OPENAI_API_KEY environment variable.")

    evaluate_performance(
        input_path=args.input,
        api_key=api_key,
        num_variants=args.num_variants
    )
