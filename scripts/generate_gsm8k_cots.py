import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"   # Stable & easy to run
OUTPUT_DIR = "./cot_output"
MAX_PROBLEMS = 200      # adjust as needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN")                     # export HF_TOKEN=xxxx
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading tokenizer/model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
# Fix tokenizer threading conflicts
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    token=HF_TOKEN
).to(DEVICE)

# -----------------------------
# HELPER: Generate CoT
# -----------------------------
def generate_cot(problem):
    prompt = f"""You are a math reasoning assistant. 
Solve the following problem using step-by-step chain-of-thought reasoning and give the final answer at the end.

Problem:
{problem}

Answer with:
<reasoning>
...step-by-step reasoning...
</reasoning>
<final>
...final numeric answer...
</final>
"""

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# -----------------------------
# MAIN SCRIPT
# -----------------------------
def main():
    print("Loading GSM8K...")
    dataset = load_dataset("gsm8k", "main")["train"]
    
    # Clear any existing cache to prevent conflicts
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    print("Starting generation...")
    for i, item in enumerate(dataset):
        if i >= MAX_PROBLEMS:
            break

        problem_text = item['question']

        out_path = f"{OUTPUT_DIR}/gsm8k_{i}.json"
        if os.path.exists(out_path):
            print(f"Skipping {i} (already generated).")
            continue

        try:
            print(f"Generating problem {i}...")
            cot = generate_cot(problem_text)

            with open(out_path, "w") as f:
                json.dump({
                    "id": i,
                    "problem": problem_text,
                    "generated_cot": cot
                }, f, indent=2)

        except Exception as e:
            print(f"Error on problem {i}: {e}")
            continue

    print("DONE.")

if __name__ == "__main__":
    main()
