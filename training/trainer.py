import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from tqdm.auto import tqdm
import os
import json
from typing import Dict, Tuple, List


class FCMTrainer:
    """
    Trainer class for FCM following README specifications
    """
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer (AdamW as specified in README)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Will be set up in train()
        self.scheduler = None
        self.scaler = None
        
        # Training tracking
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_macro_f1': [],
            'val_faithful_f1': []
        }
        
        self.best_faithful_f1 = 0.0
        self.patience_counter = 0
    
    def setup_training(self, train_dataloader, total_steps):
        """Setup scheduler and mixed precision"""
        # Learning rate scheduler with warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Mixed precision scaler (fp16)
        if self.config.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def train_epoch(self, train_dataloader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.config.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    logits = self.model(input_ids, attention_mask)
                    loss = nn.CrossEntropyLoss()(logits, labels)
            else:
                logits = self.model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(logits, labels)
            
            # Backward pass
            if self.config.use_mixed_precision:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def evaluate(self, eval_dataloader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(logits, labels)
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(eval_dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        macro_f1 = f1_score(all_labels, all_predictions, average='macro')
        
        # Critical faithfulness F1 (F vs U classification)
        faithful_labels = [1 if l < 2 else 0 for l in all_labels]  # FC,FI=1, UC,UI=0
        faithful_preds = [1 if p < 2 else 0 for p in all_predictions]
        faithful_f1 = f1_score(faithful_labels, faithful_preds)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'faithful_f1': faithful_f1,  # Critical metric from README
            'confusion_matrix': confusion_matrix(all_labels, all_predictions).tolist()
        }
    
    def save_model(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.output_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"✅ New best model saved! Faithful F1: {metrics['faithful_f1']:.4f}")
    
    def train(self, train_dataset, eval_dataset):
        """Main training loop"""
        # Create data loaders
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        eval_dataloader = DataLoader(
            eval_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False
        )
        
        # Setup training
        total_steps = len(train_dataloader) * self.config.num_epochs
        self.setup_training(train_dataloader, total_steps)
        
        print(f"🚀 Starting training for {self.config.num_epochs} epochs")
        print(f"📊 Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
        print(f"⚙️  Batch size: {self.config.batch_size}, Total steps: {total_steps}")
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            print(f"\n📅 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss, train_accuracy = self.train_epoch(train_dataloader)
            
            # Evaluate
            eval_metrics = self.evaluate(eval_dataloader)
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_accuracy)
            self.training_history['val_loss'].append(eval_metrics['loss'])
            self.training_history['val_accuracy'].append(eval_metrics['accuracy'])
            self.training_history['val_macro_f1'].append(eval_metrics['macro_f1'])
            self.training_history['val_faithful_f1'].append(eval_metrics['faithful_f1'])
            
            # Print metrics
            print(f"📈 Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            print(f"📊 Val Loss: {eval_metrics['loss']:.4f}, Val Acc: {eval_metrics['accuracy']:.4f}")
            print(f"🎯 Val Macro-F1: {eval_metrics['macro_f1']:.4f}, Faithful-F1: {eval_metrics['faithful_f1']:.4f}")
            
            # Check if best model (based on faithful F1 as specified in README)
            is_best = eval_metrics['faithful_f1'] > self.best_faithful_f1
            if is_best:
                self.best_faithful_f1 = eval_metrics['faithful_f1']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save model
            self.save_model(epoch + 1, eval_metrics, is_best)
            
            # Early stopping check
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                break
        
        print(f"🎉 Training complete! Best Faithful F1: {self.best_faithful_f1:.4f}")
        return self.training_history