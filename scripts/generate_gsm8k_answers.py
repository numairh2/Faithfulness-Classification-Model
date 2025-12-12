import json
from pathlib import Path

cot_dir = Path('cot_output')
output_file = 'data_processed/gsm8k_with_answers.jsonl'

# Read all JSON files from cot_output
results = []
for json_file in sorted(cot_dir.glob('*.json'), key=lambda x: int(x.stem.split('_')[1])):
    with open(json_file, 'r') as f:
        data = json.load(f)
        results.append(data)

print(f'Found {len(results)} CoT files')

# Now read the extracted answers
with open('extracted_answers.jsonl', 'r') as f:
    answers_data = [json.loads(line) for line in f]

print(f'Found {len(answers_data)} extracted answers')

# Merge the data
merged = []
for i, result in enumerate(results):
    if i < len(answers_data):
        result.update({
            'extracted_answer': answers_data[i].get('extracted_answer'),
            'correctness': answers_data[i].get('correctness')
        })
    merged.append(result)

# Save
with open(output_file, 'w') as f:
    for item in merged:
        f.write(json.dumps(item) + '\\n')

print(f'Saved {len(merged)} items to {output_file}')
