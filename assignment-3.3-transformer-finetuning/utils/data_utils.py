"""
Data preprocessing utilities for sentiment analysis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional


class SentimentDataset(Dataset):
    """Custom Dataset class for sentiment analysis."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_imdb_dataset() -> Tuple[Dict, Dict]:
    """Load IMDB movie review dataset."""
    dataset = load_dataset("imdb")
    
    train_data = {
        'texts': dataset['train']['text'],
        'labels': dataset['train']['label']
    }
    
    test_data = {
        'texts': dataset['test']['text'],
        'labels': dataset['test']['label']
    }
    
    return train_data, test_data


def load_csv_dataset(file_path: str, text_column: str = 'text', 
                    label_column: str = 'label') -> Dict:
    """Load dataset from CSV file."""
    df = pd.read_csv(file_path)
    
    return {
        'texts': df[text_column].tolist(),
        'labels': df[label_column].tolist()
    }


def preprocess_text(text: str) -> str:
    """Basic text preprocessing."""
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Convert to lowercase (optional - BERT is case-sensitive)
    # text = text.lower()
    
    return text


def split_dataset(texts: List[str], labels: List[int], 
                 test_size: float = 0.2, val_size: float = 0.1,
                 random_state: int = 42) -> Tuple:
    """Split dataset into train, validation, and test sets."""
    
    # First split: train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Second split: train and val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_data_loaders(train_dataset, val_dataset, test_dataset, 
                       batch_size: int = 16) -> Tuple:
    """Create DataLoaders for training, validation, and testing."""
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, test_loader


def get_class_weights(labels: List[int]) -> torch.Tensor:
    """Calculate class weights for imbalanced datasets."""
    from sklearn.utils.class_weight import compute_class_weight
    
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight(
        'balanced', classes=unique_labels, y=labels
    )
    
    return torch.tensor(class_weights, dtype=torch.float)


def analyze_dataset(texts: List[str], labels: List[int]) -> Dict:
    """Analyze dataset statistics."""
    
    stats = {
        'total_samples': len(texts),
        'num_classes': len(set(labels)),
        'class_distribution': dict(zip(*np.unique(labels, return_counts=True))),
        'avg_text_length': np.mean([len(text.split()) for text in texts]),
        'max_text_length': max([len(text.split()) for text in texts]),
        'min_text_length': min([len(text.split()) for text in texts])
    }
    
    return stats
