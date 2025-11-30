import torch
from torch.utils.data import Dataset
import json
from typing import Dict, List, Optional
from .tokenizer import FCMTokenizer

class FCMDataset(Dataset):
    """
    Dataset class for FCM training data
    
    Loads the structured FCM data created by your data processing pipeline
    """

    def __init__(self, data_file: str, tokenizer: FCMTokenizer, max_length: int = 1024):
        """
        Initialize FCM dataset
        
        Args:
            data_file: Path to JSONL file (e.g., data_processed/fcm_train.jsonl)
            tokenizer: FCMTokenizer instance
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data from JSONL file
        self.data = []
        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
        
        # Label mapping (FC=0, FI=1, UC=2, UI=3)
        self.label_to_id = {
            "FC": 0, "FI": 1, "UC": 2, "UI": 3
        }
        
        print(f"Loaded {len(self.data)} examples from {data_file}")
        
        # Verify all labels are valid
        for item in self.data:
            if item['label'] not in self.label_to_id:
                raise ValueError(f"Invalid label: {item['label']}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example
        
        Returns:
            Dictionary with input_ids, attention_mask, label, and metadata
        """
        item = self.data[idx]
        
        # Tokenize the structured FCM input text
        encoding = self.tokenizer.encode_fcm_input(
            item['input_text'], 
            max_length=self.max_length
        )
        
        # Convert label to tensor
        label = torch.tensor(self.label_to_id[item['label']], dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),  # Remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': label,
            'id': item['id'],
            'faithfulness': item['faithfulness'],
            'correctness': item['correctness']
        }

    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in the dataset"""
        from collections import Counter
        labels = [item['label'] for item in self.data]
        return dict(Counter(labels))

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalance"""
        class_counts = self.get_class_distribution()
        total_samples = len(self.data)
        
        # Calculate inverse frequency weights
        weights = []
        for label in ['FC', 'FI', 'UC', 'UI']:
            count = class_counts.get(label, 1)  # Avoid division by zero
            weight = total_samples / (len(class_counts) * count)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float)