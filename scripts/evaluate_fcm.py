#!/usr/bin/env python3
"""
Evaluate trained FCM model on test data
Usage: python scripts/evaluate_fcm.py --model models/trained_fcm/best_model.pt --test-data data_processed/fcm_test.jsonl
"""

import torch
import argparse
import json
import os
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models import create_fcm_model
from models.dataset import FCMDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class FCMEvaluator:
    """Evaluate trained FCM model"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize evaluator with trained model
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to run evaluation on
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model checkpoint
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extract config
        self.config = checkpoint['config']
        
        # Create model and tokenizer
        self.model, self.tokenizer = create_fcm_model(
            model_name=self.config.model_name,
            num_classes=self.config.num_classes,
            dropout=self.config.dropout
        )
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Label mappings
        self.id_to_label = {0: "FC", 1: "FI", 2: "UC", 3: "UI"}
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        
        print(f"✅ Model loaded successfully on {self.device}")
        print(f"📊 Model info: {self.config.model_name}, {self.config.num_classes} classes")
    
    def predict_batch(self, dataloader: DataLoader) -> Tuple[List[int], List[int], List[float]]:
        """
        Run inference on a batch of data
        
        Returns:
            predictions, true_labels, confidence_scores
        """
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                
                # Get predictions
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                confidences = torch.max(probabilities, dim=-1)[0]
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        return all_predictions, all_labels, all_confidences
    
    def calculate_metrics(self, predictions: List[int], labels: List[int]) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        macro_f1 = f1_score(labels, predictions, average='macro')
        weighted_f1 = f1_score(labels, predictions, average='weighted')
        
        # Per-class F1 scores
        per_class_f1 = f1_score(labels, predictions, average=None)
        
        # Critical faithfulness F1 (F vs U classification)
        faithful_labels = [1 if l < 2 else 0 for l in labels]  # FC,FI=1, UC,UI=0
        faithful_preds = [1 if p < 2 else 0 for p in predictions]
        faithful_f1 = f1_score(faithful_labels, faithful_preds)
        
        # Correctness F1 (C vs I classification) 
        correct_labels = [1 if l % 2 == 0 else 0 for l in labels]  # FC,UC=1, FI,UI=0
        correct_preds = [1 if p % 2 == 0 else 0 for p in predictions]
        correct_f1 = f1_score(correct_labels, correct_preds)
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'faithful_f1': faithful_f1,  # Critical metric
            'correct_f1': correct_f1,
            'fc_f1': per_class_f1[0],
            'fi_f1': per_class_f1[1],
            'uc_f1': per_class_f1[2],
            'ui_f1': per_class_f1[3]
        }
    
    def plot_confusion_matrix(self, predictions: List[int], labels: List[int], output_dir: str = None):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['FC', 'FI', 'UC', 'UI'],
            yticklabels=['FC', 'FI', 'UC', 'UI']
        )
        plt.title('FCM Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if output_dir:
            plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix saved to {output_dir}/confusion_matrix.png")
        
        plt.show()
        
        return cm
    
    def analyze_errors(self, predictions: List[int], labels: List[int], 
                      test_data: List[Dict], confidences: List[float], 
                      output_dir: str = None) -> Dict:
        """Analyze prediction errors in detail"""
        
        error_analysis = {
            'total_errors': 0,
            'faithful_errors': 0,  # F classified as U or vice versa
            'correctness_errors': 0,  # C classified as I or vice versa
            'low_confidence_errors': [],
            'high_confidence_errors': []
        }
        
        errors = []
        
        for i, (pred, label, conf) in enumerate(zip(predictions, labels, confidences)):
            if pred != label:
                error_analysis['total_errors'] += 1
                
                # Check error type
                pred_faithful = pred < 2
                true_faithful = label < 2
                pred_correct = pred % 2 == 0
                true_correct = label % 2 == 0
                
                if pred_faithful != true_faithful:
                    error_analysis['faithful_errors'] += 1
                    
                if pred_correct != true_correct:
                    error_analysis['correctness_errors'] += 1
                
                # Store error details
                error_info = {
                    'index': i,
                    'predicted': self.id_to_label[pred],
                    'actual': self.id_to_label[label],
                    'confidence': conf,
                    'question': test_data[i].get('question', '')[:200] + "...",
                    'reasoning': test_data[i].get('model_reasoning', '')[:200] + "..."
                }
                
                if conf < 0.6:
                    error_analysis['low_confidence_errors'].append(error_info)
                else:
                    error_analysis['high_confidence_errors'].append(error_info)
                
                errors.append(error_info)
        
        # Save detailed error analysis
        if output_dir:
            with open(f"{output_dir}/error_analysis.json", 'w') as f:
                json.dump(error_analysis, f, indent=2, default=str)
            print(f"📋 Error analysis saved to {output_dir}/error_analysis.json")
        
        return error_analysis
    
    def evaluate_dataset(self, test_data_path: str, output_dir: str = None) -> Dict:
        """
        Evaluate model on test dataset
        
        Args:
            test_data_path: Path to test JSONL file
            output_dir: Directory to save results
            
        Returns:
            Dictionary with all evaluation results
        """
        print(f"🔍 Loading test data from {test_data_path}")
        
        # Load test dataset
        test_dataset = FCMDataset(test_data_path, self.tokenizer, self.config.max_sequence_length)
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        print(f"📊 Test dataset: {len(test_dataset)} examples")
        print(f"📈 Class distribution: {test_dataset.get_class_distribution()}")
        
        # Run predictions
        predictions, labels, confidences = self.predict_batch(test_dataloader)
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, labels)
        
        # Load raw test data for error analysis
        test_data = []
        with open(test_data_path, 'r') as f:
            for line in f:
                test_data.append(json.loads(line))
        
        # Create output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Plot confusion matrix
        cm = self.plot_confusion_matrix(predictions, labels, output_dir)
        
        # Error analysis
        error_analysis = self.analyze_errors(predictions, labels, test_data, confidences, output_dir)
        
        # Compile results
        results = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'error_analysis': error_analysis,
            'model_info': {
                'model_path': self.model_path,
                'model_name': self.config.model_name,
                'num_classes': self.config.num_classes
            },
            'test_info': {
                'test_data_path': test_data_path,
                'num_samples': len(test_dataset),
                'class_distribution': test_dataset.get_class_distribution()
            }
        }
        
        # Save results
        if output_dir:
            with open(f"{output_dir}/evaluation_results.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Evaluation results saved to {output_dir}/evaluation_results.json")
        
        # Print summary
        self.print_results_summary(results)
        
        return results
    
    def print_results_summary(self, results: Dict):
        """Print a formatted summary of evaluation results"""
        metrics = results['metrics']
        
        print("\n" + "="*60)
        print("📊 FCM EVALUATION RESULTS")
        print("="*60)
        print(f"📈 Overall Accuracy:     {metrics['accuracy']:.4f}")
        print(f"🎯 Macro F1:            {metrics['macro_f1']:.4f}")
        print(f"⚖️  Weighted F1:         {metrics['weighted_f1']:.4f}")
        print()
        print("🔍 CRITICAL METRICS:")
        print(f"✨ Faithfulness F1:     {metrics['faithful_f1']:.4f} (MOST IMPORTANT)")
        print(f"✅ Correctness F1:      {metrics['correct_f1']:.4f}")
        print()
        print("📋 PER-CLASS F1 SCORES:")
        print(f"   FC (Faithful+Correct):   {metrics['fc_f1']:.4f}")
        print(f"   FI (Faithful+Incorrect): {metrics['fi_f1']:.4f}")
        print(f"   UC (Unfaithful+Correct): {metrics['uc_f1']:.4f}")
        print(f"   UI (Unfaithful+Incorrect): {metrics['ui_f1']:.4f}")
        print()
        print("🚨 ERROR ANALYSIS:")
        error_analysis = results['error_analysis']
        print(f"   Total errors: {error_analysis['total_errors']}")
        print(f"   Faithfulness errors: {error_analysis['faithful_errors']}")
        print(f"   Correctness errors: {error_analysis['correctness_errors']}")
        print(f"   Low confidence errors: {len(error_analysis['low_confidence_errors'])}")
        print(f"   High confidence errors: {len(error_analysis['high_confidence_errors'])}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained FCM model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test JSONL file")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Directory to save results")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.test_data):
        print(f"❌ Test data file not found: {args.test_data}")
        sys.exit(1)
    
    try:
        # Initialize evaluator
        evaluator = FCMEvaluator(args.model, args.device)
        
        # Run evaluation
        results = evaluator.evaluate_dataset(args.test_data, args.output_dir)
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📂 Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()