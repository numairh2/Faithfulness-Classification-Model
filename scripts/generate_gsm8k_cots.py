import os
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
if 'transformers' in sys.modules:
    del sys.modules['transformers']
    del sys.modules['transformers.models.auto']
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_cot(problem, tokenizer, model, device):
    """Generate chain-of-thought reasoning for a problem"""
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

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)

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


def generate_gsm8k_cots(input_file, output_file, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
    """Generate CoTs for problems in input JSONL file"""

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = os.getenv("HF_TOKEN")

    print(f"Using device: {device}")
    print("Loading tokenizer/model...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    token=hf_token,
    ).to(device)

    # Clear cache
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    # Read input problems
    print(f"Loading problems from {input_file}...")
    problems = []
    with open(input_file, 'r') as f:
        for line in f:
            problems.append(json.loads(line))

    print(f"Loaded {len(problems)} problems")

    # Check for existing output to resume
    existing_ids = set()
    if os.path.exists(output_file):
        print(f"Found existing output file, will resume...")
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                existing_ids.add(data['problem_id'])
        print(f"Already processed {len(existing_ids)} problems")

    # Generate CoTs
    print("Starting CoT generation...")
    output_mode = 'a' if existing_ids else 'w'

    with open(output_file, output_mode) as f:
        for i, problem_data in enumerate(problems):
            problem_id = problem_data['problem_id']

            # Skip if already processed
            if problem_id in existing_ids:
                print(f"Skipping problem {problem_id} (already generated)")
                continue

            try:
                print(f"Generating CoT for problem {problem_id} ({i+1}/{len(problems)})...")

                cot = generate_cot(problem_data['question'], tokenizer, model, device)

                # Write to JSONL (one line per problem)
                output_record = {
                    **problem_data,  # Include all original fields
                    'generated_cot': cot
                }
                f.write(json.dumps(output_record) + '\n')
                f.flush()  # Ensure it's written immediately

            except Exception as e:
                print(f"Error on problem {problem_id}: {e}")
                continue

    print(f"✅ CoT generation complete!")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate chain-of-thought reasoning for GSM8K problems")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with problems")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file for generated CoTs")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3",
                        help="Model name to use for generation")

    args = parser.parse_args()

    generate_gsm8k_cots(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
