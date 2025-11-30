# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a chain-of-thought (COT) reasoning research project focused on generating step-by-step solutions for GSM8K math problems. The project extracts math problems from the GSM8K dataset and generates chain-of-thought reasoning using language models.

## Architecture

The project follows a simple pipeline:
1. **Data Extraction**: `scripts/extract_gsm8k.py` samples problems from the GSM8K dataset
2. **COT Generation**: `scripts/generate_gsm8k_cots.py` generates chain-of-thought reasoning using Mistral-7B
3. **Data Storage**: Raw and processed data stored in `data_raw/` and `data_processed/` respectively

## Key Components

### Data Pipeline
- `scripts/extract_gsm8k.py`: Extracts 500 random problems from GSM8K train split into JSONL format
- `scripts/generate_gsm8k_cots.py`: Generates chain-of-thought reasoning using Mistral-7B-Instruct-v0.3
- Data flows from `data_raw/gsm8k/` → processing → `data_processed/`

### Model Configuration
- Uses Mistral-7B-Instruct-v0.3 for COT generation
- Requires HuggingFace token via `HF_TOKEN` environment variable
- GPU acceleration when available, falls back to CPU

## Common Development Commands

### Running Data Extraction
```bash
cd scripts
python extract_gsm8k.py
```

### Running COT Generation
```bash
export HF_TOKEN=your_hf_token_here
cd scripts  
python generate_gsm8k_cots.py
```

## Dependencies

The project requires:
- `datasets` (HuggingFace)
- `transformers` 
- `torch`
- Standard library: `json`, `os`, `random`

## Data Format

Input problems (GSM8K subset):
```json
{
  "problem_id": 0,
  "question": "...",
  "gold_solution": "...", 
  "gold_answer": "..."
}
```

Generated COT format uses structured prompting with `<reasoning>` and `<final>` tags.