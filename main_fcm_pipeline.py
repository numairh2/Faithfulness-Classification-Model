#!/usr/bin/env python3
"""
Complete FCM Pipeline - From GSM8K to Trained Classifier
Usage: python main_fcm_pipeline.py
"""

import argparse
import subprocess
import os
import sys
from pathlib import Path
import json
import time
from typing import Optional

class FCMPipeline:
    """Complete pipeline orchestrator for FCM training"""
    
    def __init__(self, config):
        self.config = config
        self.project_root = Path(__file__).parent
        
        # Define all paths
        self.paths = {
            'gsm8k_raw': self.project_root / 'data_raw' / 'gsm8k' / 'gsm8k_subset.jsonl',
            'gsm8k_cots': self.project_root / 'data_processed' / 'gsm8k_cots.jsonl',
            'gsm8k_answers': self.project_root / 'data_processed' / 'gsm8k_with_answers.jsonl',
            'gsm8k_labeled': self.project_root / 'data_processed' / 'gsm8k_labeled.jsonl',
            'fcm_data': self.project_root / 'data_processed' / 'fcm_data.jsonl',
            'fcm_train': self.project_root / 'data_processed' / 'fcm_train.jsonl',
            'fcm_dev': self.project_root / 'data_processed' / 'fcm_dev.jsonl',
            'fcm_test': self.project_root / 'data_processed' / 'fcm_test.jsonl',
            'trained_model': self.project_root / 'models' / 'trained_fcm' / 'best_model.pt'
        }
        
        # Ensure directories exist
        os.makedirs(self.project_root / 'data_raw' / 'gsm8k', exist_ok=True)
        os.makedirs(self.project_root / 'data_processed', exist_ok=True)
        os.makedirs(self.project_root / 'models' / 'trained_fcm', exist_ok=True)
    
    def run_command(self, cmd: list, description: str, check_output_file: Optional[Path] = None):
        """Run a subprocess command with proper error handling"""
        print(f"\n🔄 {description}")
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=self.project_root)
            print(f"✅ {description} completed successfully")
            
            if check_output_file and not check_output_file.exists():
                print(f"⚠️  Warning: Expected output file {check_output_file} not found")
                return False
                
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} failed")
            print(f"Error: {e.stderr}")
            return False
    
    def check_file_exists(self, file_path: Path, description: str) -> bool:
        """Check if a required file exists"""
        if file_path.exists():
            print(f"✅ Found: {description} ({file_path})")
            return True
        else:
            print(f"❌ Missing: {description} ({file_path})")
            return False
    
    def phase_0_data_acquisition(self) -> bool:
        """Phase 0: Extract GSM8K and generate CoTs"""
        print("\n" + "="*60)
        print("📋 PHASE 0: DATA ACQUISITION")
        print("="*60)
        
        # Step 1: Extract GSM8K subset
        if not self.paths['gsm8k_raw'].exists() or self.config.force_regenerate:
            cmd = [
                'python', 'scripts/extract_gsm8k.py',
                '--output', str(self.paths['gsm8k_raw']),
                '--num-samples', str(self.config.num_samples)
            ]
            if not self.run_command(cmd, "Extracting GSM8K subset", self.paths['gsm8k_raw']):
                return False
        else:
            print(f"✅ GSM8K subset already exists: {self.paths['gsm8k_raw']}")
        
        # Step 2: Generate CoTs
        if not self.paths['gsm8k_cots'].exists() or self.config.force_regenerate:
            cmd = [
                'python', 'scripts/generate_gsm8k_cots.py',
                '--input', str(self.paths['gsm8k_raw']),
                '--output', str(self.paths['gsm8k_cots'])
            ]
            if not self.run_command(cmd, "Generating CoT reasoning", self.paths['gsm8k_cots']):
                return False
        else:
            print(f"✅ CoT data already exists: {self.paths['gsm8k_cots']}")
        
        return True
    
    def phase_1_data_processing(self) -> bool:
        """Phase 1: Process CoT data into FCM training format"""
        print("\n" + "="*60)
        print("🔧 PHASE 1: DATA PROCESSING")
        print("="*60)
        
        # Step 3: Extract answers
        if not self.paths['gsm8k_answers'].exists() or self.config.force_regenerate:
            cmd = [
                'python', 'scripts/extract_answers.py',
                '--input', str(self.paths['gsm8k_cots']),
                '--output', str(self.paths['gsm8k_answers'])
            ]
            if not self.run_command(cmd, "Extracting final answers", self.paths['gsm8k_answers']):
                return False
        else:
            print(f"✅ Answers already extracted: {self.paths['gsm8k_answers']}")
        
        # Step 4: Interactive faithfulness labeling
        if not self.paths['gsm8k_labeled'].exists() or self.config.force_regenerate:
            print("\n🏷️  INTERACTIVE FAITHFULNESS LABELING")
            print("=" * 50)
            print("You will now label each CoT as faithful or unfaithful.")
            print("Criteria for faithfulness:")
            print("  ✅ FAITHFUL: Valid logic, correct arithmetic, no hallucinations")
            print("  ❌ UNFAITHFUL: Fabricated numbers, logical errors, inconsistencies")
            print()
            input("Press Enter to start interactive labeling...")
            
            cmd = [
                'python', 'scripts/label_faithfulness.py',
                '--input', str(self.paths['gsm8k_answers']),
                '--output', str(self.paths['gsm8k_labeled'])
            ]
            if not self.run_command(cmd, "Interactive faithfulness labeling", self.paths['gsm8k_labeled']):
                return False
        else:
            print(f"✅ Faithfulness labels already exist: {self.paths['gsm8k_labeled']}")
        
        # Step 5: Prepare FCM data
        if not self.paths['fcm_data'].exists() or self.config.force_regenerate:
            cmd = [
                'python', 'scripts/prepare_fcm_data.py',
                '--input', str(self.paths['gsm8k_labeled']),
                '--output', str(self.paths['fcm_data'])
            ]
            if not self.run_command(cmd, "Preparing FCM training data", self.paths['fcm_data']):
                return False
        else:
            print(f"✅ FCM data already prepared: {self.paths['fcm_data']}")
        
        # Step 6: Create dataset splits
        if not all([self.paths['fcm_train'].exists(), self.paths['fcm_dev'].exists(), self.paths['fcm_test'].exists()]) or self.config.force_regenerate:
            cmd = [
                'python', 'scripts/create_dataset_splits.py',
                '--input', str(self.paths['fcm_data']),
                '--output-dir', str(self.project_root / 'data_processed')
            ]
            if not self.run_command(cmd, "Creating train/dev/test splits", self.paths['fcm_train']):
                return False
        else:
            print(f"✅ Dataset splits already exist")
        
        return True
    
    def phase_2_training(self) -> bool:
        """Phase 2: Train the FCM model"""
        print("\n" + "="*60)
        print("🚀 PHASE 2: MODEL TRAINING")
        print("="*60)
        
        # Check that training data exists
        if not all([self.paths['fcm_train'].exists(), self.paths['fcm_dev'].exists()]):
            print("❌ Training data not found. Run data processing first.")
            return False
        
        # Train model
        cmd = [
            'python', 'training/train_fcm.py',
            '--train-data', str(self.paths['fcm_train']),
            '--dev-data', str(self.paths['fcm_dev']),
            '--output-dir', str(self.project_root / 'models' / 'trained_fcm'),
            '--epochs', str(self.config.epochs),
            '--batch-size', str(self.config.batch_size),
            '--learning-rate', str(self.config.learning_rate)
        ]
        
        if not self.run_command(cmd, "Training FCM model", self.paths['trained_model']):
            return False
        
        return True
    
    def phase_3_evaluation(self) -> bool:
        """Phase 3: Evaluate the trained model"""
        print("\n" + "="*60)
        print("📊 PHASE 3: MODEL EVALUATION")
        print("="*60)
        
        # Check that model and test data exist
        if not self.paths['trained_model'].exists():
            print("❌ Trained model not found. Run training first.")
            return False
        
        if not self.paths['fcm_test'].exists():
            print("❌ Test data not found. Run data processing first.")
            return False
        
        # Evaluate model
        cmd = [
            'python', 'scripts/evaluate_fcm.py',
            '--model', str(self.paths['trained_model']),
            '--test-data', str(self.paths['fcm_test'])
        ]
        
        if not self.run_command(cmd, "Evaluating trained model"):
            return False
        
        return True
    
    def run_full_pipeline(self) -> bool:
        """Run the complete FCM pipeline"""
        start_time = time.time()
        
        print("🎯 FCM COMPLETE PIPELINE")
        print("=" * 60)
        print(f"Starting from: {self.config.start_from}")
        print(f"Number of samples: {self.config.num_samples}")
        print(f"Force regenerate: {self.config.force_regenerate}")
        print("=" * 60)
        
        # Run phases based on starting point
        if self.config.start_from in ['gsm8k', 'data']:
            if not self.phase_0_data_acquisition():
                return False
        
        if self.config.start_from in ['gsm8k', 'data', 'processing']:
            if not self.phase_1_data_processing():
                return False
        
        if self.config.start_from in ['gsm8k', 'data', 'processing', 'training']:
            if not self.phase_2_training():
                return False
        
        if self.config.start_from in ['gsm8k', 'data', 'processing', 'training', 'evaluation']:
            if not self.phase_3_evaluation():
                return False
        
        # Pipeline complete
        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print("🎉 FCM PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"⏱️  Total time: {elapsed/60:.1f} minutes")
        print(f"📂 Trained model: {self.paths['trained_model']}")
        print(f"📊 Training data: {self.paths['fcm_train']}")
        print(f"🔬 Test data: {self.paths['fcm_test']}")
        print("="*60)
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Complete FCM Pipeline - From GSM8K to Trained Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_fcm_pipeline.py                    # Run complete pipeline
  python main_fcm_pipeline.py --start-from processing  # Skip data generation
  python main_fcm_pipeline.py --num-samples 500        # Use fewer samples
        """
    )
    
    parser.add_argument(
        '--start-from', 
        choices=['gsm8k', 'data', 'processing', 'training', 'evaluation'],
        default='gsm8k',
        help='Pipeline starting point (default: gsm8k)'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='Number of GSM8K samples to process (default: 1000)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs (default: 3)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Training batch size (default: 8)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='Learning rate (default: 2e-5)'
    )
    
    parser.add_argument(
        '--force-regenerate',
        action='store_true',
        help='Force regeneration of existing files'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = FCMPipeline(args)
    
    try:
        success = pipeline.run_full_pipeline()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()