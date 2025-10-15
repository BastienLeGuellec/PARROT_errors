#!/usr/bin/env python3
"""
Evaluates an LLM's ability to detect laterality errors in reports.

This script takes an input file containing original reports and versions
with introduced laterality errors.

It iterates through each text, asks a specified LLM (e.g., GPT-4o) to find
the single incorrect laterality word, and compares the LLM's prediction to the
ground truth.

Finally, it calculates and plots performance metrics:
- Sensitivity (Recall)
- Specificity
- F1 Score

Dependencies:
    pip install pandas openai matplotlib scikit-learn

CLI Example:
    python spot_laterality_errors.py \
        --input reports_with_laterality_errors.jsonl \
        --api-key "YOUR_OPENAI_API_KEY"
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
    Asks the LLM to find a single incorrect laterality word in the text.

    Returns:
        The incorrect word or "N/A" if no error is found.
    """
    prompt = f"""You are an expert medical proofreader with a very high standard for accuracy. Your task is to identify ONLY clear, unambiguous laterality errors (e.g., 'left' incorrectly used instead of 'right') in the following medical report.

- A laterality error is a factual contradiction where 'left' should be 'right', or 'right' should be 'left'. Do not flag stylistic issues.
- If and only if you find a clear laterality error, return the single incorrect word (e.g., 'left' or 'right').
- If and only if there are no such errors, you MUST return the exact string 'N/A'. 

Now, analyze the following text:
Text:
{text}"""

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
            prediction = prediction.strip('.,;:"\'').lower()
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
    return word.strip('.,;:"\'').lower()

# --- Main Evaluation Logic ---

def evaluate_performance(input_path: str, api_key: str):
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

        # --- Evaluate Negative Cases (Originals for Specificity) ---
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
                
                if pred_result == 1: # Log incorrect predictions (False Positives)
                    print("--- INCORRECT PREDICTION (False Positive) ---")
                    print(f"Report (Correct): {text[:300]}...")
                    print(f"Prediction: {prediction}")
                    print(f"Expected: N/A")
                    print("-------------------------------------------")

                results[current_lang]['y_pred'].append(pred_result)
                results['overall']['y_pred'].append(pred_result)

        # --- Evaluate Positive Cases (with errors for Sensitivity) ---
        error_texts = {
            "report_with_error": (row.get("report_with_error"), lang),
            "translation_with_error": (row.get("translation_with_error"), 'english')
        }
        for col_name, (error_text, current_lang) in error_texts.items():
            if pd.notna(error_text):
                if current_lang not in results:
                    results[current_lang] = {'y_true': [], 'y_pred': []}
                results[current_lang]['y_true'].append(1) # Ground truth: Error exists
                results['overall']['y_true'].append(1)
                
                prediction = get_llm_prediction(client, error_text)
                
                # If the LLM returns anything other than "N/A", it's a success.
                pred_result = 1 if prediction != "N/A" else 0
                
                if pred_result == 0: # Log incorrect predictions (False Negatives)
                    print("--- INCORRECT PREDICTION (False Negative) ---")
                    print(f"Report with error: {error_text[:300]}...")
                    print(f"Prediction: {prediction}")
                    print(f"Expected: Any word (not 'N/A')")
                    print("-------------------------------------------")

                results[current_lang]['y_pred'].append(pred_result)
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
        plt.title('Overall LLM Performance in Laterality Error Detection')
        plt.ylim(0, 1.1)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('llm_laterality_performance.png')
        print("\nOverall performance graph saved to llm_laterality_performance.png")


# --- CLI ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on laterality error detection.")
    parser.add_argument("--input", type=str, required=True, help="Path to input data file (.jsonl or .xlsx).")
    parser.add_argument("--api-key", type=str, help="OpenAI API key. Can also be set via OPENAI_API_KEY environment variable.")
    
    args = parser.parse_args()

    api_key = "sk-sdU_3qoZa-TBKfMRIGGugsmCLkhNZFUi9QpFubFgTST3BlbkFJ2bYHLsOUeRcmSdcTOYPVHujcfuGfzRLiFqPoLNEX4A"

    if not api_key:
        raise ValueError("OpenAI API key must be provided via --api-key argument or OPENAI_API_KEY environment variable.")

    evaluate_performance(
        input_path=args.input,
        api_key=api_key
    )
