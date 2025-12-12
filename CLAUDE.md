# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Faithfulness Classification Model (FCM)** project that trains a DeBERTa-v3-small classifier to evaluate Chain-of-Thought (CoT) reasoning quality. The model classifies CoT outputs into four categories based on faithfulness (reasoning integrity) and correctness (answer accuracy).

**Key Classification Labels:**
- **FC**: Faithful + Correct
- **FI**: Faithful + Incorrect
- **UC**: Unfaithful + Correct
- **UI**: Unfaithful + Incorrect

This is NOT a generative model - it judges the quality of existing CoT reasoning.

## Environment Setup

**Python Version:** 3.9 (required for PyTorch 1.9.0 compatibility)

**Create Environment:**
```bash
conda create -n fcm python=3.9 -y
conda activate fcm
```

**Install Dependencies:**
```bash
# PyTorch 1.9.0 with CUDA 11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Core packages
pip install transformers==4.36.0 sentencepiece protobuf
pip install datasets accelerate bitsandbytes
pip install scikit-learn numpy pandas tqdm matplotlib seaborn

# For notebook support
pip install jupyter ipykernel ipywidgets
python -m ipykernel install --user --name=fcm --display-name="Python 3.9 (FCM)"
```

**Verify Installation:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Common Commands

### Run Complete Pipeline
```bash
# Full pipeline from data extraction to trained model
python main_fcm_pipeline.py

# Use fewer samples for testing
python main_fcm_pipeline.py --num-samples 100

# Force regenerate existing files
python main_fcm_pipeline.py --force-regenerate
```

### Resume from Specific Phase
```bash
# Skip data generation, start from processing
python main_fcm_pipeline.py --start-from processing

# Skip to training (if data is ready)
python main_fcm_pipeline.py --start-from training

# Only run evaluation
python main_fcm_pipeline.py --start-from evaluation
```

### Custom Training Parameters
```bash
python main_fcm_pipeline.py --epochs 5 --batch-size 16 --learning-rate 1e-5
```

### Manual Evaluation
```bash
python scripts/evaluate_fcm.py \
    --model models/trained_fcm/best_model.pt \
    --test-data data_processed/fcm_test.jsonl
```

### Check GPU Status
```bash
nvidia-smi
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

## Architecture Overview

### Pipeline Flow

The project follows a sequential 4-phase pipeline:

**Phase 0: Data Acquisition**
- `scripts/extract_gsm8k.py`: Samples problems from GSM8K dataset
- `scripts/generate_gsm8k_cots.py`: Generates CoT reasoning using Mistral-7B-Instruct-v0.3
- Output: `data_raw/gsm8k/gsm8k_subset.jsonl` → `data_processed/gsm8k_cots.jsonl`

**Phase 1: Data Processing**
- `scripts/extract_answers.py`: Extracts final numerical answers from CoT outputs
- `scripts/label_faithfulness.py`: **Interactive manual labeling** of faithfulness
- `scripts/prepare_fcm_data.py`: Converts to structured FCM format
- `scripts/create_dataset_splits.py`: Creates train/dev/test splits (70%/15%/15%)
- Output: `data_processed/fcm_{train,dev,test}.jsonl`

**Phase 2: Model Training**
- `training/train_fcm.py`: Main training script
- Uses `training/trainer.py` for training loop
- Configuration via `training/config.py`
- Output: `models/trained_fcm/best_model.pt`

**Phase 3: Evaluation**
- `scripts/evaluate_fcm.py`: Comprehensive model evaluation
- Metrics: Accuracy, Macro-F1, **Faithful-F1** (critical metric)
- Generates confusion matrix and error analysis

### Model Architecture

**Base Model:** `microsoft/deberta-v3-small` (DeBERTa-v3 encoder)

**Classification Head:**
```
DeBERTa Encoder (768-dim)
    ↓
[CLS] Token Pooling
    ↓
Linear(768 → 256)
    ↓
GELU Activation
    ↓
Dropout(0.1)
    ↓
Linear(256 → 4)
    ↓
4 Class Logits (FC/FI/UC/UI)
```

Defined in `models/fcm.py:FaithfulnessClassifier`

### Input Format

The model expects structured text input (created by `scripts/prepare_fcm_data.py`):

```
[QUESTION]
{original GSM8K question}

[GROUND_TRUTH]
{gold solution or numeric answer}

[MODEL_REASONING]
{model's full chain of thought text}

[MODEL_ANSWER]
{model's final answer or parsed integer}
```

This structured format ensures the classifier has all necessary context without needing to infer missing information.

### Key Components

**`main_fcm_pipeline.py`**
- Orchestrates entire pipeline via `FCMPipeline` class
- Handles phase transitions and error recovery
- Validates intermediate outputs exist before proceeding

**`models/fcm.py`**
- Core `FaithfulnessClassifier(nn.Module)` implementation
- Methods: `forward()`, `predict()`, `get_faithfulness_score()`
- Label mappings: FC=0, FI=1, UC=2, UI=3

**`models/tokenizer.py`**
- `FCMTokenizer` wrapper around HuggingFace tokenizer
- Handles structured input encoding with 1024 max tokens
- Batch encoding support

**`models/dataset.py`**
- `FCMDataset(Dataset)` for PyTorch DataLoader
- Loads JSONL data files
- Provides class distribution and balancing weights

**`training/config.py`**
- `FCMTrainingConfig` dataclass with all hyperparameters
- Defaults match README specifications (lr=2e-5, epochs=2-3, etc.)

**`training/trainer.py`**
- `FCMTrainer` class implementing training loop
- AdamW optimizer, mixed precision (fp16)
- Early stopping based on **faithful-F1** score (not overall accuracy)

## Important Implementation Details

### Faithfulness Labeling

During Phase 1, the pipeline pauses for **interactive manual labeling**. Criteria:

**Faithful (F):**
- Valid logical reasoning
- Correct arithmetic operations
- No hallucinated numbers
- Consistent step-by-step logic

**Unfaithful (U):**
- Fabricated or incorrect numbers
- Mathematical errors
- Logical inconsistencies
- Unexplained reasoning jumps

### Training Specifications

**Loss Function:** CrossEntropyLoss over 4 classes

**Optimizer:** AdamW
- Learning rate: 2e-5
- Weight decay: 0.01

**Training Parameters:**
- Batch size: 8-16
- Epochs: 2-3
- Max sequence length: 1024 tokens (CoTs are long!)
- Mixed precision: fp16
- Early stopping: Based on faithful-F1 (not accuracy)

**Critical Metric:** Faithful-F1 score (measures faithfulness detection quality)
- More important than overall accuracy
- Model checkpointing uses this metric

### Data Format Standards

**GSM8K Subset (`data_raw/gsm8k/gsm8k_subset.jsonl`):**
```json
{
  "problem_id": 0,
  "question": "...",
  "gold_solution": "...",
  "gold_answer": "..."
}
```

**FCM Training Data (`data_processed/fcm_train.jsonl`):**
```json
{
  "id": 0,
  "input_text": "[QUESTION]\n...\n[GROUND_TRUTH]\n...",
  "label": "FC",
  "faithfulness": "F",
  "correctness": "C",
  "metadata": {...}
}
```

### CoT Generation

**Model:** Mistral-7B-Instruct-v0.3 (not part of training - just data generation)
- Requires `HF_TOKEN` environment variable
- Uses structured prompt with `<reasoning>` and `<final>` tags
- GPU-accelerated when available
- Outputs saved to `cot_output/gsm8k_{id}.json`

## Directory Structure

```
Faithfulness-Classification-Model/
├── main_fcm_pipeline.py          # Pipeline orchestrator
├── scripts/                      # Data processing pipeline
│   ├── extract_gsm8k.py         # Phase 0: Extract GSM8K subset
│   ├── generate_gsm8k_cots.py   # Phase 0: Generate CoT reasoning
│   ├── extract_answers.py       # Phase 1: Extract final answers
│   ├── label_faithfulness.py    # Phase 1: Interactive labeling
│   ├── prepare_fcm_data.py      # Phase 1: Format for FCM
│   ├── create_dataset_splits.py # Phase 1: Train/dev/test splits
│   └── evaluate_fcm.py          # Phase 3: Model evaluation
├── models/                       # Model architecture
│   ├── fcm.py                   # FaithfulnessClassifier
│   ├── tokenizer.py             # FCMTokenizer wrapper
│   ├── dataset.py               # FCMDataset class
│   └── __init__.py              # Model factory functions
├── training/                     # Training infrastructure
│   ├── config.py                # FCMTrainingConfig
│   ├── trainer.py               # FCMTrainer training loop
│   └── train_fcm.py             # Main training script
├── data_raw/                     # Raw data (auto-created)
├── data_processed/               # Processed data (auto-created)
├── cot_output/                   # Generated CoTs (auto-created)
└── models/trained_fcm/           # Saved models (auto-created)
```

## Performance Expectations

### Training Time
- 100 samples: ~5-10 minutes
- 1000 samples: ~30-60 minutes
- 5000+ samples: ~2-4 hours

### Expected Metrics
- **Accuracy**: 70-85%
- **Faithful-F1** (critical): 75-90%
- **Macro-F1**: 65-80%

Lower performance indicates insufficient training data or poor labeling quality.

## Troubleshooting

### CUDA/GPU Issues
```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU if needed
export CUDA_VISIBLE_DEVICES=""
```

### Memory Issues
```bash
# Reduce batch size
python main_fcm_pipeline.py --batch-size 4
```

### Pipeline Recovery
```bash
# Verify intermediate outputs exist
ls -la data_processed/
wc -l data_processed/*.jsonl

# Resume from last successful phase
python main_fcm_pipeline.py --start-from processing
```

### HuggingFace Token Required
```bash
# For CoT generation with Mistral
export HF_TOKEN=your_hf_token_here
```

## Development Notes

- **Do NOT modify** the structured input format in `prepare_fcm_data.py` - the model expects exact `[QUESTION]`, `[GROUND_TRUTH]`, `[MODEL_REASONING]`, `[MODEL_ANSWER]` markers
- The **faithful-F1 score** is the primary metric for model selection, not overall accuracy
- CoT generation (Phase 0) is separate from FCM training - FCM only judges existing CoTs
- Manual labeling quality in Phase 1 directly impacts final model performance
- The 1024 token limit is critical for long CoT reasoning - do not reduce it
- Class imbalance is common (UC examples are rare) - dataset provides class weights
