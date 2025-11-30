import re
import json
import os
from pathlib import Path


def extract_final_answer(cot_text):
    """Extract numeric answer from CoT text using multiple patterns"""
    if not cot_text:
        return None
    
    # Common patterns in math reasoning (ordered by priority)
    patterns = [
        r'#### (\d+(?:\.\d+)?)',                           # GSM8K format: #### 42
        r'<final>\s*(\d+(?:\.\d+)?)\s*</final>',          # XML tags: <final>42</final>
        r'[Tt]he answer is (\d+(?:\.\d+)?)',              # "The answer is 42"
        r'[Ff]inal answer:\s*(\d+(?:\.\d+)?)',            # "Final answer: 42"
        r'[Ss]o the answer is (\d+(?:\.\d+)?)',           # "So the answer is 42"
        r'[Tt]herefore,?\s*(\d+(?:\.\d+)?)',              # "Therefore, 42"
        r'= (\d+(?:\.\d+)?)(?:\s*$|\s*\.$)',              # "= 42" at end
        r'\$(\d+(?:\.\d+)?)',                             # Dollar amounts: $42
        r'(\d+(?:\.\d+)?)\s*(?:dollars?|cents?)',         # "42 dollars"
        r'answer:\s*(\d+(?:\.\d+)?)',                     # "answer: 42"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, cot_text, re.MULTILINE | re.IGNORECASE)
        if matches:
            try:
                # Take the last match (most likely to be final answer)
                return float(matches[-1])
            except (ValueError, IndexError):
                continue
    
    # Fallback: look for any number at the end
    end_number = re.search(r'(\d+(?:\.\d+)?)\s*$', cot_text.strip())
    if end_number:
        try:
            return float(end_number.group(1))
        except ValueError:
            pass
    
    return None

def extract_gold_answer(gold_text):
    """Extract numeric answer from gold solution text"""
    if isinstance(gold_text, (int, float)):
        return float(gold_text)

    if not isinstance(gold_text, str):
        return None
    
    # GSM8K gold answers often end with #### number
    gold_match = re.search(r'#### (\d+(?:\.\d+)?)', gold_text)
    if gold_match:
        return float(gold_match.group(1))
    
    # Fallback to general extraction
    return extract_final_answer(gold_text)

def check_correctness(extracted_answer, gold_answer, tolerance=0.01):
    """Check if extracted answer matches gold answer"""
    if extracted_answer is None:
        return "I"  # Incorrect (couldn't extract)
    
    gold_num = extract_gold_answer(gold_answer)
    if gold_num is None:
        return "I"  # Couldn't parse gold answer
    
    # Check if answers match within tolerance
    return "C" if abs(extracted_answer - gold_num) <= tolerance else "I"

def process_cot_file(file_path):
    """Process a single CoT JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract answer from generated CoT
        extracted_answer = extract_final_answer(data.get('generated_cot', ''))

        # Check correctness
        gold_answer = data.get('gold_answer', data.get('answer', ''))
        correctness = check_correctness(extracted_answer, gold_answer)

        # Add to data
        data['extracted_answer'] = extracted_answer
        data['correctness'] = correctness

        return data

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    
def extract_answers_from_directory(input_dir="cot_output", output_file="extracted_answers.jsonl"):
    """Process all CoT files and extract answers"""
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Directory {input_dir} not found!")
        return

    results = []
    json_files = list(input_path.glob("*.json"))

    print(f"Processing {len(json_files)} CoT files...")

    for file_path in json_files:
        result = process_cot_file(file_path)
        if result:
            results.append(result)

    # Save results
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Statistics
    total = len(results)
    extracted = sum(1 for r in results if r.get('extracted_answer') is not None)
    correct = sum(1 for r in results if r.get('correctness') == 'C')

    print(f"\nResults:")
    print(f"Total examples: {total}")
    print(f"Answers extracted: {extracted} ({extracted/total*100:.1f}%)")
    print(f"Correct answers: {correct} ({correct/total*100:.1f}%)")
    print(f"Saved to: {output_file}")

    return results


if __name__ == "__main__":
    extract_answers_from_directory()
