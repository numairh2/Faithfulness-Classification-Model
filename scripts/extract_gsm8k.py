from datasets import load_dataset
import json
import os
import random
import argparse


def extract_gsm8k_subset(output_file, num_samples=500):
    """Extract a random subset of GSM8K problems"""

    # Create output directory
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load GSM8K Dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")
    all_examples = dataset["train"]

    # Sample random subset
    actual_num = min(num_samples, len(all_examples))
    print(f"Sampling {actual_num} problems from {len(all_examples)} total...")

    sampled_examples = random.sample(list(all_examples), actual_num)

    # Write to JSONL
    with open(output_file, "w") as f:
        for idx, ex in enumerate(sampled_examples):
            record = {
                "problem_id": idx,
                "question": ex["question"].strip(),
                "gold_solution": ex["answer"].strip(),
                "gold_answer": ex["answer"].strip(),
            }
            f.write(json.dumps(record) + "\n")

    print(f"✅ GSM8K subset saved: {output_file}")
    print(f"Total problems: {actual_num}")

    return actual_num


def main():
    parser = argparse.ArgumentParser(description="Extract random subset of GSM8K dataset")
    parser.add_argument("--output", type=str, default="data_raw/gsm8k/gsm8k_subset.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--num-samples", type=int, default=500,
                        help="Number of problems to sample")

    args = parser.parse_args()

    extract_gsm8k_subset(args.output, args.num_samples)


if __name__ == "__main__":
    main()
