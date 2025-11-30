#!/usr/bin/env python3
"""
Main training script for FCM
Usage: python training/train_fcm.py
"""

import torch
import argparse
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_fcm_model
from models.dataset import FCMDataset
from training.config import FCMTrainingConfig
from training.trainer import FCMTrainer


def main():
    parser = argparse.ArgumentParser(description="Train FCM model")
    parser.add_argument("--config", type=str, help="Path to config file (optional)")
    parser.add_argument("--train-data", type=str, default="data_processed/fcm_train.jsonl")
    parser.add_argument("--dev-data", type=str, default="data_processed/fcm_dev.jsonl")
    parser.add_argument("--output-dir", type=str, default="models/trained_fcm")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    
    args = parser.parse_args()
    
    # Create config
    config = FCMTrainingConfig(
        train_data_file=args.train_data,
        dev_data_file=args.dev_data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print(f"=' Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Output dir: {config.output_dir}")
    
    # Create model and tokenizer
    print(f"> Loading model and tokenizer...")
    model, tokenizer = create_fcm_model(
        model_name=config.model_name,
        num_classes=config.num_classes,
        dropout=config.dropout
    )
    
    # Load datasets
    print(f"=Ú Loading datasets...")
    train_dataset = FCMDataset(config.train_data_file, tokenizer, config.max_sequence_length)
    eval_dataset = FCMDataset(config.dev_data_file, tokenizer, config.max_sequence_length)
    
    # Print dataset info
    print(f"=Ê Train dataset distribution: {train_dataset.get_class_distribution()}")
    print(f"=Ê Eval dataset distribution: {eval_dataset.get_class_distribution()}")
    
    # Create trainer
    trainer = FCMTrainer(model, tokenizer, config)
    
    # Train model
    history = trainer.train(train_dataset, eval_dataset)
    
    print(f" Training complete!")
    print(f"=È Final metrics saved to: {config.output_dir}")


if __name__ == "__main__":
    main()