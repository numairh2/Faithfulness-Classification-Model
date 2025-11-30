from datasets import load_dataset
import json
import os
import random


# Configurations

OUTPUT_DIR = "data_raw/gsm8k"
OUTPUT_FILE = "gsm8k_subset.jsonl"
NUM_PROBLEMS = 500

# Create Output Folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load GSM8K Dataset

dataset = load_dataset("gsm8k", "main")  # "main" split

# Use the train split
all_examples = dataset["train"]

# Sample Random Subset

if NUM_PROBLEMS > len(all_examples):
    NUM_PROBLEMS = len(all_examples)

sampled_examples = random.sample(list(all_examples), NUM_PROBLEMS)

# Write to JSONL

output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

with open(output_path, "w") as f:
    for idx, ex in enumerate(sampled_examples):
        record = {
            "problem_id": idx,
            "question": ex["question"].strip(),
            "gold_solution": ex["answer"].strip(),
            "gold_answer": ex["answer"].strip(),  # GSM8K has "answer" field
        }
        f.write(json.dumps(record) + "\n")

print(f"✅ GSM8K subset saved: {output_path}")
print(f"Total problems: {NUM_PROBLEMS}")



