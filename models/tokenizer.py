from transformers import AutoTokenizer
import torch
from typing import List, Dict, Union

class FCMTokenizer:
    """
    Tokenizer wrapper for FCM that handles the structured input format
    """

    def __init__(self, model_name="microsoft/deberta-v3-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = 1024  # As specified in README
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode_fcm_input(self, fcm_text: str, max_length: int = None) -> Dict[str, torch.Tensor]:
        """
        Encode FCM structured input text
        
        Args:
            fcm_text: Structured input with [QUESTION], [GROUND_TRUTH], etc.
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        max_len = max_length or self.max_length
        
        encoding = self.tokenizer(
            fcm_text,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        )
        
        return encoding

    def encode_batch(self, fcm_texts: List[str], max_length: int = None) -> Dict[str, torch.Tensor]:
        """
        Encode batch of FCM inputs
        
        Args:
            fcm_texts: List of structured input texts
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with batched tensors
        """
        max_len = max_length or self.max_length
        
        encoding = self.tokenizer(
            fcm_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        )
        
        return encoding

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """Decode token IDs back to text"""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)