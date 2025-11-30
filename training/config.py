from dataclasses import dataclass
from typing import Optional

@dataclass
class FCMTrainingConfig:
    """Training configuration for FCM as specified in README"""

    # Model settings
    model_name: str = "microsoft/deberta-v3-small"
    num_classes: int = 4
    dropout: float = 0.1
    max_sequence_length: int = 1024

    # Training parameters (from README)
    batch_size: int = 8  # README: 8-16
    num_epochs: int = 3  # README: 2-3 epochs
    learning_rate: float = 2e-5  # README specification
    weight_decay: float = 0.01   # README specification

    # Optimization settings
    warmup_steps: int = 100
    max_grad_norm: float = 1.0  # Gradient clipping
    use_mixed_precision: bool = True  # fp16

    # Data paths
    train_data_file: str = "data_processed/fcm_train.jsonl"
    dev_data_file: str = "data_processed/fcm_dev.jsonl"
    test_data_file: str = "data_processed/fcm_test.jsonl"

    # Training settings
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50
    save_total_limit: int = 3

    # Output settings
    output_dir: str = "models/trained_fcm"
    run_name: str = "fcm_training"

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001