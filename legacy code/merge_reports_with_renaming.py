import json
import os

def merge_jsonl_files(files, output_file):
    merged_data = {}

    for file_path in files:
        filename = os.path.basename(file_path)
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    key = data.get("no")
                    if key is None:
                        continue

                    try:
                        key = int(key)
                    except (ValueError, TypeError):
                        print(f"Skipping line with invalid 'no' key in {filename}: {line.strip()}")
                        continue

                    if key not in merged_data:
                        merged_data[key] = {}

                    # Rename keys for laterality and negation files
                    if "laterality" in filename:
                        rename_map = {
                            'report_with_error': 'report_with_laterality_error',
                            'translation_with_error': 'translation_with_laterality_error',
                            'original_mistake': 'original_laterality_mistake',
                            'corrected_mistake': 'corrected_laterality_mistake',
                            'explanation': 'laterality_explanation',
                            'category': 'laterality_category'
                        }
                        for old, new in rename_map.items():
                            if old in data:
                                data[new] = data.pop(old)
                    elif "negation" in filename:
                        rename_map = {
                            'report_with_error': 'report_with_negation_error',
                            'translation_with_error': 'translation_with_negation_error',
                            'original_mistake': 'original_negation_mistake',
                            'corrected_mistake': 'corrected_negation_mistake',
                            'explanation': 'negation_explanation',
                            'category': 'negation_category'
                        }
                        for old, new in rename_map.items():
                            if old in data:
                                data[new] = data.pop(old)

                    merged_data[key].update(data)

                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in {filename}: {line.strip()}")

    with open(output_file, 'w') as f:
        for key in sorted(merged_data.keys()):
            f.write(json.dumps(merged_data[key]) + '\n')

if __name__ == "__main__":
    files_to_merge = [
        "reports_dmeta.jsonl",
        "reports_with_negation_errors.jsonl",
        "reports_with_laterality_errors.jsonl"
    ]
    output_filename = "merged_reports.jsonl"

    merge_jsonl_files(files_to_merge, output_filename)

    print(f"Successfully merged {len(files_to_merge)} files into {output_filename} with corrected logic.")
