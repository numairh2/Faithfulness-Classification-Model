import json
import os
from pathlib import Path
import re
from collections import Counter
import argparse


def clean_text(text):
    """Clean and normalize text"""
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\.\,\!\?\$\%\(\)\-\+\=\<\>\/\:]', '', text)

    return text   

def create_fcm_format(question, gold_answer, cot_text, extracted_answer):
    """Convert to README's required structured format"""
    
    # Clean inputs
    question = clean_text(question)
    gold_answer = clean_text(str(gold_answer))
    cot_text = clean_text(cot_text)
    
    # Format extracted answer
    if extracted_answer is not None:
        extracted_str = str(extracted_answer)
    else:
        extracted_str = "N/A"
    
    # Create structured format as specified in README
    formatted_input = f"""[QUESTION]
{question}

[GROUND_TRUTH]
{gold_answer}

[MODEL_REASONING]
{cot_text}

[MODEL_ANSWER]
{extracted_str}"""
    
    return formatted_input
def validate_example(example):
    """Validate that example has required fields"""
    required_fields = ['id', 'problem', 'generated_cot', 'faithfulness', 'correctness']
    
    for field in required_fields:
        if field not in example:
            return False, f"Missing field: {field}"
    
    # Check label validity
    if example['faithfulness'] not in ['F', 'U']:
        return False, f"Invalid faithfulness: {example['faithfulness']}"
    
    if example['correctness'] not in ['C', 'I']:
        return False, f"Invalid correctness: {example['correctness']}"
    
    return True, "Valid"

def prepare_fcm_dataset(input_file="labeled_data.jsonl", output_dir="data_processed"):
    """Prepare dataset in FCM format with preprocessing"""
    
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found!")
        print("Run label_faithfulness.py first to create labeled data.")
        return
    
    # Load labeled data
    raw_data = []
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line)
                raw_data.append(item)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {line_num}: {e}")
    
    print(f"Loaded {len(raw_data)} labeled examples")
    
    # Validate and process data
    fcm_data = []
    invalid_count = 0
    
    for item in raw_data:
        is_valid, error_msg = validate_example(item)
        if not is_valid:
            print(f"Skipping invalid example {item.get('id', 'unknown')}: {error_msg}")
            invalid_count += 1
            continue
        
        # Create FCM format
        try:
            fcm_input = create_fcm_format(
                item['problem'],
                item.get('gold_answer', ''),
                item['generated_cot'],
                item.get('extracted_answer', None)
            )
            
            # Remove overly short CoTs as specified in README
            if len(item['generated_cot'].split()) < 15:
                print(f"Skipping short CoT for example {item['id']}")
                continue
            
            fcm_example = {
                'id': item['id'],
                'input_text': fcm_input,
                'label': item['final_label'],
                'faithfulness': item['faithfulness'],
                'correctness': item['correctness'],
                'metadata': {
                    'original_problem': item['problem'],
                    'gold_answer': item.get('gold_answer', ''),
                    'extracted_answer': item.get('extracted_answer', None),
                    'cot_length': len(item['generated_cot'])
                }
            }
            
            fcm_data.append(fcm_example)
            
        except Exception as e:
            print(f"Error processing example {item.get('id', 'unknown')}: {e}")
            invalid_count += 1
    
    print(f"Successfully processed {len(fcm_data)} examples")
    print(f"Invalid/skipped: {invalid_count} examples")
    
    if len(fcm_data) == 0:
        print("No valid examples to process!")
        return
    
    # Check class distribution
    label_dist = Counter([item['label'] for item in fcm_data])
    
    print(f"\n📊 Class Distribution:")
    for label in ['FC', 'FI', 'UC', 'UI']:
        count = label_dist.get(label, 0)
        percentage = count / len(fcm_data) * 100 if len(fcm_data) > 0 else 0
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Check for class imbalance
    min_class_count = min(label_dist.values()) if label_dist else 0
    max_class_count = max(label_dist.values()) if label_dist else 0
    if max_class_count > 0 and min_class_count / max_class_count < 0.1:
        print("⚠️  Warning: Significant class imbalance detected!")
        print("   Consider collecting more examples for rare classes.")
    
    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/fcm_processed.jsonl"
    
    with open(output_file, 'w') as f:
        for item in fcm_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n✅ Saved {len(fcm_data)} processed examples to {output_file}")
    
    # Save summary statistics
    summary = {
        'total_examples': len(fcm_data),
        'class_distribution': dict(label_dist),
        'processing_stats': {
            'input_examples': len(raw_data),
            'valid_examples': len(fcm_data),
            'invalid_examples': invalid_count
        },
        'data_quality': {
            'avg_cot_length': sum(item['metadata']['cot_length'] for item in fcm_data) / len(fcm_data),
            'min_cot_length': min(item['metadata']['cot_length'] for item in fcm_data),
            'max_cot_length': max(item['metadata']['cot_length'] for item in fcm_data)
        }
    }
    
    summary_file = f"{output_dir}/processing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📈 Saved processing summary to {summary_file}")
    
    return fcm_data


def main():
    parser = argparse.ArgumentParser(description="Prepare FCM dataset")
    parser.add_argument("--input", default="labeled_data.jsonl", help="Input labeled data file")
    parser.add_argument("--output-dir", default="data_processed", help="Output directory")
    
    args = parser.parse_args()
    
    prepare_fcm_dataset(args.input, args.output_dir)


if __name__ == "__main__":
    main()