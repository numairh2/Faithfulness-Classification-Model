# FCM Pipeline - Complete Setup and Usage Guide

This document provides step-by-step instructions for running the complete Faithfulness Classification Model (FCM) pipeline from raw GSM8K data to trained classifier.

## 📋 Overview

The FCM pipeline automatically handles:
- **Data Acquisition**: Extract GSM8K subset and generate Chain-of-Thought reasoning
- **Data Processing**: Extract answers, label faithfulness, prepare training data
- **Model Training**: Train DeBERTa-v3-small classifier on FCM task
- **Evaluation**: Comprehensive model testing and performance analysis


### Run Complete Pipeline
```bash
python main_fcm_pipeline.py
```

This single command will execute the entire pipeline from start to finish.

## 🚀 Initial System Setup

### Step 1: NVIDIA Driver Installation (For GPU Support)

**Check if drivers are already installed:**
```bash
nvidia-smi
```

If this command works, you already have drivers installed. Otherwise:

**Ubuntu/Debian:**
```bash
# Update package list
sudo apt update

# Install NVIDIA driver (recommended version)
sudo apt install nvidia-driver-535

# Reboot system
sudo reboot

# Verify installation
nvidia-smi
```

**Alternative: Install latest driver from NVIDIA**
```bash
# Add NVIDIA PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Find recommended driver
ubuntu-drivers devices

# Install recommended driver (replace XXX with version number)
sudo apt install nvidia-driver-XXX
```

### Step 2: Conda Environment Setup

**Option A: Install Conda (if not already installed)**

```bash
# Download Anaconda installer
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh

# Run installer
bash Anaconda3-2022.05-Linux-x86_64.sh

# Follow prompts, then initialize conda
source ~/.bashrc

# Verify installation
conda --version
```

**Option B: Use Miniconda (lighter alternative)**
```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install
bash Miniconda3-latest-Linux-x86_64.sh

# Initialize
source ~/.bashrc
```

### Step 3: Create Conda Environment with Python 3.9

**Why Python 3.9?** The notebook uses PyTorch 1.9.0 which requires Python 3.9 for compatibility.

```bash
# Create conda environment with Python 3.9
conda create -n fcm python=3.9 -y

# Activate environment
conda activate fcm

# Verify Python version
python --version  # Should show Python 3.9.x
```

### Step 4: Install PyTorch 1.9.0 with CUDA 11.1

```bash
# Install PyTorch 1.9.0 with CUDA 11.1 support
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 5: Install Core Dependencies

```bash
# Install transformers (compatible version for PyTorch 1.9)
pip install transformers==4.36.0

# Install tokenizer dependencies
pip install sentencepiece protobuf

# Install datasets and acceleration libraries
pip install datasets accelerate bitsandbytes

# Install ML and visualization libraries
pip install scikit-learn numpy pandas tqdm matplotlib seaborn

# For Jupyter notebook support
pip install jupyter ipykernel ipywidgets

# Register kernel for Jupyter
python -m ipykernel install --user --name=fcm --display-name="Python 3.9 (FCM)"
```

### Step 6: Verify Installation

```bash
# Test all critical imports
python << EOF
import torch
import transformers
import datasets
import sentencepiece
import sklearn
print("✅ All core packages installed successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF
```

## 📋 Quick Start After Setup

Once your environment is configured:

```bash
# Activate conda environment
conda activate fcm

# Run the pipeline
python main_fcm_pipeline.py
```

## 🔧 Environment Management

**Activate environment:**
```bash
conda activate fcm
```

**Deactivate environment:**
```bash
conda deactivate
```

**Update packages:**
```bash
conda activate fcm
pip install --upgrade transformers datasets
```

**Remove environment (if needed):**
```bash
conda deactivate
conda env remove -n fcm
```

**Export environment for sharing:**
```bash
conda activate fcm
conda env export > environment.yml
```

**Recreate from environment file:**
```bash
conda env create -f environment.yml
```

## 📋 Package Version Reference

For reproducibility, here are the exact versions used:

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.9.25 | Base interpreter |
| PyTorch | 1.9.0+cu111 | Deep learning framework |
| transformers | 4.36.0 | Hugging Face models |
| tokenizers | 0.15.2 | Fast tokenization |
| sentencepiece | 0.2.1 | Tokenizer backend |
| protobuf | 6.33.1 | Serialization |
| datasets | 4.4.1 | Dataset loading |
| accelerate | 1.10.1 | Training acceleration |
| bitsandbytes | 0.48.2 | Quantization support |
| scikit-learn | latest | ML utilities |
| pandas | latest | Data manipulation |
| numpy | 2.0.2 | Numerical computing |

### Project Structure
Ensure your project has this structure:
```
COT/
├── main_fcm_pipeline.py          # Main pipeline script
├── scripts/                      # Data processing scripts
│   ├── extract_gsm8k.py
│   ├── generate_gsm8k_cots.py
│   ├── extract_answers.py
│   ├── label_faithfulness.py
│   ├── prepare_fcm_data.py
│   ├── create_dataset_splits.py
│   └── evaluate_fcm.py
├── models/                       # Model architecture
│   ├── fcm.py
│   ├── tokenizer.py
│   ├── dataset.py
│   └── __init__.py
├── training/                     # Training infrastructure
│   ├── config.py
│   ├── trainer.py
│   └── train_fcm.py
├── data_raw/                     # Raw data (auto-created)
├── data_processed/               # Processed data (auto-created)
└── models/trained_fcm/           # Saved models (auto-created)
```

## 🔧 Pipeline Phases

### Phase 0: Data Acquisition
- **Extract GSM8K subset** from dataset
- **Generate CoT reasoning** using your model

### Phase 1: Data Processing  
- **Extract final answers** from CoT outputs
- **Interactive faithfulness labeling** (manual step)
- **Prepare FCM training data** in structured format
- **Create train/dev/test splits** (70%/15%/15%)

### Phase 2: Model Training
- **Train DeBERTa-v3-small** classifier
- **AdamW optimizer** with specified hyperparameters
- **Mixed precision training** (fp16)
- **Early stopping** based on faithful-F1 score

### Phase 3: Evaluation
- **Comprehensive metrics** (accuracy, F1 scores)
- **Confusion matrix** visualization
- **Error analysis** with confidence scores

## 📝 Usage Examples

### 1. Complete Pipeline (Recommended)
```bash
# Run everything from scratch
python main_fcm_pipeline.py

# Use fewer samples for testing
python main_fcm_pipeline.py --num-samples 100

# Force regenerate existing files
python main_fcm_pipeline.py --force-regenerate
```

### 2. Resume from Specific Phase
```bash
# Skip data generation, start from processing
python main_fcm_pipeline.py --start-from processing

# Skip to training (if data is ready)
python main_fcm_pipeline.py --start-from training

# Only run evaluation
python main_fcm_pipeline.py --start-from evaluation
```

### 3. Custom Training Parameters
```bash
# Adjust training hyperparameters
python main_fcm_pipeline.py --epochs 5 --batch-size 16 --learning-rate 1e-5

# Combine with other options
python main_fcm_pipeline.py --start-from training --epochs 3 --batch-size 8
```

### 4. Manual Evaluation
```bash
# Evaluate a specific model
python scripts/evaluate_fcm.py \
    --model models/trained_fcm/best_model.pt \
    --test-data data_processed/fcm_test.jsonl \
    --output-dir evaluation_results
```

## 🎛️ Command Line Options

### Main Pipeline (`main_fcm_pipeline.py`)

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--start-from` | `gsm8k`, `data`, `processing`, `training`, `evaluation` | `gsm8k` | Pipeline starting point |
| `--num-samples` | Integer | `1000` | Number of GSM8K samples to process |
| `--epochs` | Integer | `3` | Number of training epochs |
| `--batch-size` | Integer | `8` | Training batch size |
| `--learning-rate` | Float | `2e-5` | Learning rate for AdamW optimizer |
| `--force-regenerate` | Flag | `False` | Force regeneration of existing files |

### Evaluation Script (`scripts/evaluate_fcm.py`)

| Option | Description |
|--------|-------------|
| `--model` | Path to trained model checkpoint (required) |
| `--test-data` | Path to test JSONL file (required) |
| `--output-dir` | Directory to save evaluation results |
| `--device` | Device to use (`cuda`/`cpu`) |

## 🔍 Interactive Faithfulness Labeling

During the pipeline, you'll be prompted to manually label CoT reasoning as faithful or unfaithful:

### Faithfulness Criteria
✅ **FAITHFUL**: 
- Valid logical reasoning
- Correct arithmetic operations  
- No hallucinated numbers
- Consistent step-by-step logic

❌ **UNFAITHFUL**:
- Fabricated or wrong numbers
- Mathematical errors
- Logical inconsistencies
- Unexplained jumps in reasoning

### Labeling Process
1. Pipeline pauses at labeling phase
2. Interactive prompt shows each CoT
3. You classify as `F` (faithful) or `U` (unfaithful)
4. Process continues after all examples labeled

## 📊 Expected Outputs

### Data Files Generated
```
data_processed/
├── gsm8k_cots.jsonl              # Generated CoT reasoning
├── gsm8k_with_answers.jsonl      # CoTs with extracted answers
├── gsm8k_labeled.jsonl           # Faithfulness-labeled data
├── fcm_data.jsonl                # Structured FCM format
├── fcm_train.jsonl               # Training split (70%)
├── fcm_dev.jsonl                 # Validation split (15%)
└── fcm_test.jsonl                # Test split (15%)
```

### Model Outputs
```
models/trained_fcm/
├── best_model.pt                 # Best model (highest faithful-F1)
├── checkpoint_epoch_1.pt         # Epoch checkpoints
├── checkpoint_epoch_2.pt
└── checkpoint_epoch_3.pt
```

### Evaluation Results
```
evaluation_results/               # (if running evaluation manually)
├── evaluation_results.json      # Comprehensive metrics
├── confusion_matrix.png         # Visualization
└── error_analysis.json          # Detailed error breakdown
```

## 🐛 Troubleshooting

### Common Issues

**1. CUDA/GPU Issues**
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Force CPU if needed
export CUDA_VISIBLE_DEVICES=""
python main_fcm_pipeline.py
```

**2. Memory Issues**
```bash
# Reduce batch size
python main_fcm_pipeline.py --batch-size 4

# Use fewer samples for testing
python main_fcm_pipeline.py --num-samples 100
```

**3. Missing Dependencies**
```bash
# Install missing packages
pip install transformers torch scikit-learn matplotlib seaborn tqdm
```

**4. File Permission Issues**
```bash
# Ensure script is executable
chmod +x main_fcm_pipeline.py

# Check directory permissions
ls -la data_processed/ models/
```

### Pipeline Recovery

**If pipeline fails mid-execution:**
```bash
# Resume from last successful phase
python main_fcm_pipeline.py --start-from processing  # or training/evaluation

# Force regenerate if files are corrupted
python main_fcm_pipeline.py --force-regenerate
```

**Check intermediate outputs:**
```bash
# Verify data files exist and aren't empty
ls -la data_processed/
wc -l data_processed/*.jsonl

# Check model training logs
ls -la models/trained_fcm/
```

## 📈 Performance Expectations

### Training Time
- **Small dataset (100 samples)**: ~5-10 minutes
- **Medium dataset (1000 samples)**: ~30-60 minutes  
- **Large dataset (5000+ samples)**: ~2-4 hours

### Expected Metrics
- **Accuracy**: 70-85%
- **Faithful-F1** (critical): 75-90%
- **Macro-F1**: 65-80%

Lower performance may indicate:
- Insufficient training data
- Poor faithfulness labeling quality
- Need for hyperparameter tuning

## 🎯 Next Steps After Training

1. **Analyze Results**: Review evaluation metrics and error analysis
2. **Improve Labeling**: Re-label ambiguous cases if performance is low
3. **Generate More Data**: Increase training data size for better performance
4. **Deploy Model**: Use trained classifier for CoT evaluation in other projects
5. **Experiment**: Try different hyperparameters or model architectures

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Ensure project structure matches requirements
4. Review console output for specific error messages

---

🎉 **Happy Training!** The FCM pipeline should handle everything automatically. Just run `python main_fcm_pipeline.py` and follow the prompts.