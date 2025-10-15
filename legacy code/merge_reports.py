
import json

def merge_jsonl_files(files, output_file):
    merged_data = {}

    for file in files:
        with open(file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    key = data.get("no")
                    if key is not None:
                        if key not in merged_data:
                            merged_data[key] = {}
                        merged_data[key].update(data)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in {file}: {line.strip()}")

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

    print(f"Successfully merged {len(files_to_merge)} files into {output_filename}")
