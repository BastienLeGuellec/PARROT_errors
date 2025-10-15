import pandas as pd
import json
import os
from collections import OrderedDict

def _calculate_metrics(df_group):
    """Calculates overall performance metrics for a given dataframe group."""
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for _, row in df_group.iterrows():
        error_type = row['error_type']
        detector_response = row.get('detector_response', {})
        judge_score = row.get('judge_score')

        if isinstance(detector_response, str):
            try:
                detector_response = json.loads(detector_response.replace("'", '"'))
            except json.JSONDecodeError:
                detector_response = {}

        error_detected = detector_response.get('error_detected', False)

        if error_type != 'none':  # Ground truth: Positive (error exists)
            if error_detected and judge_score == 1:
                tp += 1
            else:
                fn += 1
        else:  # Ground truth: Negative (no error)
            if error_detected:
                fp += 1
            else:
                tn += 1

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    return OrderedDict([
        ("TP", tp),
        ("FP", fp),
        ("TN", tn),
        ("FN", fn),
        ("Sensitivity", sensitivity),
        ("Specificity", specificity),
        ("Precision", precision),
        ("F1-Score", f1_score)
    ])

def _calculate_task_metrics(df_group):
    """Calculates per-task performance metrics."""
    task_metrics = {}
    for error_type, error_group in df_group.groupby('error_type'):
        
        def get_error_detected(response):
            if isinstance(response, str):
                try:
                    response = json.loads(response.replace("'", '"'))
                except json.JSONDecodeError:
                    return False
            return response.get('error_detected', False)

        if error_type == 'none':
            tn_task = 0
            fp_task = 0
            for _, row in error_group.iterrows():
                if not get_error_detected(row.get('detector_response', {})):
                    tn_task += 1
                else:
                    fp_task += 1
            specificity_task = tn_task / (tn_task + fp_task) if (tn_task + fp_task) > 0 else 0
            task_metrics['No Error'] = {'Specificity': specificity_task, 'TN': tn_task, 'FP': fp_task}
        else:
            tp_task = 0
            fn_task = 0
            for _, row in error_group.iterrows():
                if get_error_detected(row.get('detector_response', {})) and row['judge_score'] == 1:
                    tp_task += 1
                else:
                    fn_task += 1
            sensitivity_task = tp_task / (tp_task + fn_task) if (tp_task + fn_task) > 0 else 0
            task_metrics[error_type] = {'Sensitivity': sensitivity_task, 'TP': tp_task, 'FN': fn_task}
    return task_metrics

def calculate_performance(results_path: str):
    """
    Calculates and prints diagnostic performance metrics from evaluation results,
    with paired comparison for languages and their English translations.
    """
    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
        print(f"Error: The file {results_path} is empty or does not exist. Please run the evaluation script first.")
        return

    records = []
    with open(results_path, 'r') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # This handles the case where multiple JSON objects are on the same line
                try:
                    # Attempt to parse it as a stream of JSON objects
                    decoder = json.JSONDecoder()
                    pos = 0
                    while pos < len(line.strip()):
                        obj, pos = decoder.raw_decode(line.strip(), pos)
                        records.append(obj)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line.strip()}")

    if not records:
        print("No valid JSON records found in the results file.")
        return

    df = pd.DataFrame(records)

    # Map report_no to its original language, assuming 'english' is the only translation language
    report_no_to_lang = {
        row['report_no']: row['language']
        for _, row in df[df['language'] != 'english'].iterrows()
    }

    def get_comparison_language(row):
        if row['language'] != 'english':
            return row['language']
        
        original_lang = report_no_to_lang.get(row['report_no'])
        return f"{original_lang}-en" if original_lang else 'english'

    df['comparison_language'] = df.apply(get_comparison_language, axis=1)
    
    languages = set(df['comparison_language'].unique())
    pairs = []
    singles = set(languages)

    base_languages = sorted([lang for lang in languages if lang and not lang.endswith('-en')])

    for lang in base_languages:
        en_lang_equivalent = f"{lang}-en"
        if en_lang_equivalent in languages:
            pairs.append((lang, en_lang_equivalent))
            singles.discard(lang)
            singles.discard(en_lang_equivalent)

    if pairs:
        print("--- Paired Comparison of Performance (Original vs. English) ---")
        for lang1, lang2 in pairs:
            print(f"\n--- Comparison for {lang1} vs {lang2} ---")
            
            lang1_df = df[df['comparison_language'] == lang1]
            lang2_df = df[df['comparison_language'] == lang2]
            
            metrics1 = _calculate_metrics(lang1_df)
            metrics2 = _calculate_metrics(lang2_df)
            
            print(f"| Metric        | {lang1:<10} | {lang2:<10} |")
            print(f"|---------------|------------|------------|")
            for metric_name in metrics1.keys():
                val1 = metrics1[metric_name]
                val2 = metrics2[metric_name]
                if isinstance(val1, float):
                    print(f"| {metric_name:<13} | {val1:<10.2f} | {val2:<10.2f} |")
                else:
                    print(f"| {metric_name:<13} | {val1:<10} | {val2:<10} |")
            print(f"---------------------------------------------")

            print(f"\n--- Performance by Task for {lang1} vs {lang2} ---")
            task_metrics1 = _calculate_task_metrics(lang1_df)
            task_metrics2 = _calculate_task_metrics(lang2_df)
            all_tasks = sorted(set(task_metrics1.keys()) | set(task_metrics2.keys()))

            for task in all_tasks:
                print(f"  Task: {task}")
                m1 = task_metrics1.get(task, {})
                m2 = task_metrics2.get(task, {})
                if task == 'No Error':
                    print(f"    {lang1:<10}: Specificity: {m1.get('Specificity', 0):.2f} (TN: {m1.get('TN', 0)}, FP: {m1.get('FP', 0)})")
                    print(f"    {lang2:<10}: Specificity: {m2.get('Specificity', 0):.2f} (TN: {m2.get('TN', 0)}, FP: {m2.get('FP', 0)})")
                else:
                    print(f"    {lang1:<10}: Sensitivity: {m1.get('Sensitivity', 0):.2f} (TP: {m1.get('TP', 0)}, FN: {m1.get('FN', 0)})")
                    print(f"    {lang2:<10}: Sensitivity: {m2.get('Sensitivity', 0):.2f} (TP: {m2.get('TP', 0)}, FN: {m2.get('FN', 0)})")

    if singles:
        print("\n\n--- Individual Performance for Other Languages ---")
        for language in sorted(list(singles)):
            print(f"\n--- Overall Performance for {language} ---")
            lang_group = df[df['comparison_language'] == language]
            
            metrics = _calculate_metrics(lang_group)
            
            print(f"| Metric        | Value      |")
            print(f"|---------------|------------|")
            for name, value in metrics.items():
                if isinstance(value, float):
                    print(f"| {name:<13} | {value:<10.2f} |")
                else:
                    print(f"| {name:<13} | {value:<10} |")
            print(f"-------------------------------------")

            print(f"\n--- Performance by Task for {language} ---")
            task_metrics = _calculate_task_metrics(lang_group)
            for error_type, values in sorted(task_metrics.items()):
                if error_type == 'No Error':
                    print(f"  Task: No Error")
                    print(f"    Specificity: {values['Specificity']:.2f} (TN: {values['TN']}, FP: {values['FP']})")
                else:
                    print(f"  Task: {error_type}")
                    print(f"    Sensitivity: {values['Sensitivity']:.2f} (TP: {values['TP']}, FN: {values['FN']})")

if __name__ == "__main__":
    calculate_performance("evaluation_results.jsonl")