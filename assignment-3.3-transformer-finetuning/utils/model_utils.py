"""
Model utilities for transformer fine-tuning.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, Optional


class SentimentClassifier(nn.Module):
    """Transformer-based sentiment classifier."""
    
    def __init__(self, model_name: str, num_classes: int = 2, dropout: float = 0.3):
        super(SentimentClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        
        # Initialize classifier weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


def load_model_and_tokenizer(model_name: str, num_classes: int = 2) -> tuple:
    """Load pre-trained model and tokenizer."""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    model = SentimentClassifier(model_name, num_classes)
    
    return model, tokenizer


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters in model."""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params
    }


def freeze_transformer_layers(model: SentimentClassifier, 
                             num_layers_to_freeze: int) -> None:
    """Freeze the first N transformer layers."""
    
    # Freeze embeddings
    for param in model.transformer.embeddings.parameters():
        param.requires_grad = False
    
    # Freeze specified number of encoder layers
    if hasattr(model.transformer, 'encoder'):
        for i in range(min(num_layers_to_freeze, len(model.transformer.encoder.layer))):
            for param in model.transformer.encoder.layer[i].parameters():
                param.requires_grad = False


def save_model(model: SentimentClassifier, tokenizer, save_path: str) -> None:
    """Save model and tokenizer."""
    
    # Save model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model.model_name,
        'num_classes': model.num_classes
    }, f"{save_path}/model.pth")
    
    # Save tokenizer
    tokenizer.save_pretrained(f"{save_path}/tokenizer")


def load_model(model_path: str, tokenizer_path: str) -> tuple:
    """Load saved model and tokenizer."""
    
    # Load model checkpoint
    checkpoint = torch.load(f"{model_path}/model.pth")
    
    # Recreate model
    model = SentimentClassifier(
        checkpoint['model_name'], 
        checkpoint['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"{tokenizer_path}/tokenizer")
    
    return model, tokenizer


def get_model_memory_usage(model: nn.Module) -> Dict[str, float]:
    """Calculate model memory usage."""
    
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    
    return {
        'parameters_mb': param_size / 1024**2,
        'buffers_mb': buffer_size / 1024**2,
        'total_mb': size_mb
    }
