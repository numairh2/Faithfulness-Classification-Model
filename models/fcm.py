import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, List, Optional, Tuple


class FaithfulnessClassifier(nn.Module):
    """
    Faithfulness Classification Model (FCM) using DeBERTa-v3-small
    
    Classifies CoT reasoning into 4 classes:
    - FC: Faithful + Correct
    - FI: Faithful + Incorrect
    - UC: Unfaithful + Correct
    - UI: Unfaithful + Incorrect
    """

    def __init__(self, model_name="microsoft/deberta-v3-small", num_classes=4, dropout=0.1):
        super().__init__()
        
        # Load DeBERTa encoder as specified in README
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from config
        hidden_size = self.config.hidden_size  # 768 for deberta-v3-small
        
        # Classification head: 768 -> 256 -> 4 (as specified in README)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Label mappings as specified in README
        self.label_to_id = {
            "FC": 0,  # Faithful + Correct
            "FI": 1,  # Faithful + Incorrect
            "UC": 2,  # Unfaithful + Correct
            "UI": 3   # Unfaithful + Incorrect
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
        # Initialize classifier weights
        self._init_weights()
    def _init_weights(self):
        """Initialize classification head weights"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass through the model
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len] (optional for DeBERTa)
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Use [CLS] token representation (first token)
        pooled_output = encoder_outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        
        # Classification
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]
        
        return logits
    def predict(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Generate predictions with class labels and probabilities
        
        Returns:
            predictions: Predicted class IDs
            probabilities: Class probabilities
            labels: Predicted class labels
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, token_type_ids)
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            # Convert to labels
            labels = [self.id_to_label[pred.item()] for pred in predictions]
            
        return predictions, probabilities, labels
    def get_faithfulness_score(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Get faithfulness probability (F vs U classification)
        
        Returns:
            faithfulness_prob: Probability of being faithful (FC + FI vs UC + UI)
        """
        _, probabilities, _ = self.predict(input_ids, attention_mask, token_type_ids)
        
        # Sum probabilities for faithful classes (FC + FI)
        faithful_prob = probabilities[:, 0] + probabilities[:, 1]  # FC + FI
        
        return faithful_prob
