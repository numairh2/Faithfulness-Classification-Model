from .fcm import FaithfulnessClassifier
from .tokenizer import FCMTokenizer

def create_fcm_model(model_name="microsoft/deberta-v3-small", 
                     num_classes=4, 
                     dropout=0.1):
    """
    Factory function to create FCM model and tokenizer
    
    Returns:
        model: FaithfulnessClassifier instance
        tokenizer: FCMTokenizer instance
    """
    model = FaithfulnessClassifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout
    )
    
    tokenizer = FCMTokenizer(model_name=model_name)
    
    return model, tokenizer

# Export main classes
__all__ = ['FaithfulnessClassifier', 'FCMTokenizer', 'create_fcm_model']