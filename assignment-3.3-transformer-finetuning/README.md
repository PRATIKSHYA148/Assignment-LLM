# Assignment 3.3: Transformer Fine-tuning for Sentiment Analysis

## Overview
This assignment demonstrates how to fine-tune pre-trained transformer models for sentiment analysis tasks. We implement a complete pipeline from data preprocessing to model evaluation, showcasing the power of transfer learning with transformers.

## Objectives
- Fine-tune a pre-trained transformer model (BERT/RoBERTa) for sentiment classification
- Implement custom PyTorch Dataset and DataLoader for text data
- Use Hugging Face Transformers library for model loading and tokenization
- Apply proper training techniques including learning rate scheduling
- Evaluate model performance with comprehensive metrics
- Visualize training progress and results

## Features
- **Pre-trained Model Integration**: Uses BERT/RoBERTa from Hugging Face
- **Custom Dataset Implementation**: Flexible dataset class for text classification
- **Advanced Training Loop**: Includes validation, early stopping, and progress tracking
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, and confusion matrix
- **Memory Management**: Efficient GPU memory usage with gradient accumulation
- **Visualization**: Training curves, confusion matrix, and performance metrics
- **Model Persistence**: Save and load fine-tuned models

## Directory Structure
```
assignment-3.3-transformer-finetuning/
├── README.md
├── requirements.txt
├── fine-tune-transformer-for-sentimental-analysis.ipynb  # Main notebook
├── models/                           # Saved models directory
│   ├── best_model.pth
│   └── tokenizer/
├── data/                            # Dataset files
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── results/                         # Training results and plots
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── classification_report.txt
└── utils/
    ├── __init__.py
    ├── data_utils.py               # Data preprocessing utilities
    ├── model_utils.py              # Model architecture utilities
    └── training_utils.py           # Training helper functions
```

## Installation
```bash
pip install -r requirements.txt
```

## Requirements
- Python 3.8+
- PyTorch 1.9+
- Transformers 4.0+
- Datasets
- Scikit-learn
- Matplotlib
- Seaborn
- Pandas
- NumPy

## Usage

### Jupyter Notebook
Open `fine-tune-transformer-for-sentimental-analysis.ipynb` to run the complete pipeline:

1. **Data Loading**: Load and explore sentiment analysis datasets
2. **Preprocessing**: Tokenization and data preparation
3. **Model Setup**: Load pre-trained transformer and add classification head
4. **Training**: Fine-tune the model with validation monitoring
5. **Evaluation**: Comprehensive performance analysis
6. **Inference**: Test the model on new examples

### Key Components

#### 1. Dataset Class
```python
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        # Custom dataset for sentiment analysis
```

#### 2. Model Architecture
```python
class SentimentClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.3):
        # Transformer with classification head
```

#### 3. Training Loop
```python
def train_model(model, train_loader, val_loader, epochs=3):
    # Complete training pipeline with validation
```

## Model Configuration
- **Base Model**: BERT-base-uncased or RoBERTa-base
- **Sequence Length**: 128 tokens (configurable)
- **Batch Size**: 16 (adjustable based on GPU memory)
- **Learning Rate**: 2e-5 with linear warmup
- **Epochs**: 3-5 (with early stopping)
- **Dropout**: 0.3 for regularization

## Datasets Supported
- **IMDB Movie Reviews**: Binary sentiment classification
- **Stanford Sentiment Treebank (SST)**: Fine-grained sentiment analysis
- **Amazon Product Reviews**: Multi-domain sentiment analysis
- **Custom CSV**: Text and label columns

## Performance Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and macro-averaged metrics
- **Confusion Matrix**: Detailed classification breakdown
- **Training Curves**: Loss and accuracy over epochs
- **Inference Time**: Model speed analysis

## Training Features
- **Gradient Accumulation**: Handle larger effective batch sizes
- **Learning Rate Scheduling**: Warmup and linear decay
- **Early Stopping**: Prevent overfitting
- **Checkpointing**: Save best model states
- **Memory Optimization**: Efficient GPU usage
- **Mixed Precision**: Faster training with AMP

## Results Visualization
The notebook generates several visualizations:
- Training and validation loss curves
- Accuracy progression over epochs
- Confusion matrix heatmap
- Classification report summary
- Sample predictions with confidence scores

## Example Output
```
Epoch 1/3:
Train Loss: 0.542, Train Acc: 73.2%
Val Loss: 0.398, Val Acc: 82.1%

Epoch 2/3:
Train Loss: 0.312, Train Acc: 86.7%
Val Loss: 0.276, Val Acc: 88.9%

Final Test Accuracy: 89.3%
F1-Score: 0.891
```

## Model Inference
After training, the model can be used for inference:
```python
def predict_sentiment(text, model, tokenizer):
    # Returns sentiment prediction and confidence
```

## Advanced Features
- **Hyperparameter Tuning**: Grid search for optimal parameters
- **Data Augmentation**: Text augmentation techniques
- **Ensemble Methods**: Combine multiple model predictions
- **Cross-Validation**: Robust performance evaluation
- **Transfer Learning**: Adapt to new domains

## Troubleshooting
- **GPU Memory Issues**: Reduce batch size or use gradient accumulation
- **Slow Training**: Enable mixed precision or use smaller models
- **Poor Performance**: Increase training data or adjust hyperparameters
- **Overfitting**: Increase dropout or reduce learning rate

## Extensions
- Try different transformer architectures (RoBERTa, DistilBERT, ELECTRA)
- Experiment with different classification heads
- Implement attention visualization
- Add text preprocessing techniques
- Explore domain adaptation methods

## References
- [Transformers Documentation](https://huggingface.co/transformers/)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Fine-tuning Best Practices](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
