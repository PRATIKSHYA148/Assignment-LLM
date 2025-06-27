"""
Training utilities for transformer fine-tuning.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop


def setup_optimizer_and_scheduler(model, train_dataloader, 
                                 learning_rate: float = 2e-5, 
                                 num_epochs: int = 3,
                                 warmup_steps: Optional[int] = None):
    """Setup optimizer and learning rate scheduler."""
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Scheduler
    total_steps = len(train_dataloader) * num_epochs
    if warmup_steps is None:
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    return optimizer, scheduler


def train_epoch(model, train_loader, optimizer, scheduler, device, 
                criterion=None) -> Tuple[float, float]:
    """Train model for one epoch."""
    
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    for batch in train_loader:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy


def evaluate_model(model, data_loader, device, criterion=None) -> Tuple[float, float, List, List]:
    """Evaluate model on validation/test set."""
    
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy, predictions, true_labels


def train_model(model, train_loader, val_loader, device, num_epochs: int = 3,
                learning_rate: float = 2e-5, save_path: str = None) -> Dict:
    """Complete training loop with validation."""
    
    # Setup training components
    optimizer, scheduler = setup_optimizer_and_scheduler(
        model, train_loader, learning_rate, num_epochs
    )
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=2)
    
    # Move model to device
    model.to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': []
    }
    
    best_val_acc = 0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, criterion
        )
        
        # Validation
        val_loss, val_acc, _, _ = evaluate_model(
            model, val_loader, device, criterion
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epochs'].append(epoch + 1)
        
        # Print progress
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path:
                torch.save(model.state_dict(), f"{save_path}/best_model.pth")
        
        # Early stopping
        if early_stopping(val_acc):
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        print("-" * 50)
    
    return history


def plot_training_curves(history: Dict, save_path: str = None) -> None:
    """Plot training and validation curves."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(history['epochs'], history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(history['epochs'], history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(history['epochs'], history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(history['epochs'], history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/training_curves.png", dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(y_true: List, y_pred: List, 
                         class_names: List[str] = None, 
                         save_path: str = None) -> None:
    """Plot confusion matrix."""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(f"{save_path}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    
    plt.show()


def generate_classification_report(y_true: List, y_pred: List, 
                                 class_names: List[str] = None,
                                 save_path: str = None) -> str:
    """Generate detailed classification report."""
    
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    print("Classification Report:")
    print(report)
    
    if save_path:
        with open(f"{save_path}/classification_report.txt", 'w') as f:
            f.write(report)
    
    return report


def predict_sentiment(text: str, model, tokenizer, device, max_length: int = 128) -> Dict:
    """Predict sentiment for a single text."""
    
    model.eval()
    
    # Tokenize text
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities[0].cpu().numpy()
    }
