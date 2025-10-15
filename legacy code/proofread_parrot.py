import json
from spellchecker import SpellChecker
import os
from openpyxl import Workbook

def proofread_text(spell, text, log_writer):
    words = text.split()
    misspelled = spell.unknown(words)
    corrected_text = ""
    for word in words:
        if word in misspelled:
            corrected_word = spell.correction(word)
            if corrected_word is not None:
                corrected_text += corrected_word + " "
                log_writer.append([word, corrected_word])
            else:
                corrected_text += word + " "
        else:
            corrected_text += word + " "
    return corrected_text.strip()

def proofread_jsonl(input_file, output_file, log_file):
    print(f"Starting proofreading of {input_file}")
    print(f"Output will be saved to {output_file}")
    print(f"Log will be saved to {log_file}")
    spell = SpellChecker()

    # Workbook for logging
    wb = Workbook()
    ws = wb.active
    ws.title = "Corrections"
    ws.append(["Original Word", "Corrected Word"])


    # Get the absolute paths
    input_file_abs = os.path.abspath(input_file)
    output_file_abs = os.path.abspath(output_file)
    log_file_abs = os.path.abspath(log_file)

    print(f"Absolute input file path: {input_file_abs}")
    print(f"Absolute output file path: {output_file_abs}")
    print(f"Absolute log file path: {log_file_abs}")

    try:
        with open(input_file_abs, 'r') as f_in, open(output_file_abs, 'w') as f_out:
            for i, line in enumerate(f_in):
                data = json.loads(line)
                if 'text' in data:
                    data['text'] = proofread_text(spell, data['text'], ws)
                f_out.write(json.dumps(data) + '\n')
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1} lines.")
        wb.save(log_file_abs)
        print("Proofreading finished successfully.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_abs}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    input_file = 'PARROT_v1_0.jsonl'
    output_file = 'proofread_parrot.jsonl'
    log_file = 'proofread_log.xlsx'
    proofread_jsonl(input_file, output_file, log_file)
