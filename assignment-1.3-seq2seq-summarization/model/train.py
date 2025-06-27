import os
import json
import numpy as np
from seq2seq_model import Seq2SeqSummarizer
from data_processor import DataProcessor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    """Load and preprocess the dataset."""
    texts = []
    summaries = []
    
    # Load data from JSON files
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            with open(os.path.join(data_dir, filename), 'r') as f:
                data = json.load(f)
                texts.append(data['text'])
                summaries.append(data['summary'])
    
    return texts, summaries

def plot_training_history(history):
    """Plot training and validation metrics."""
    # Plot loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Parameters
    data_dir = '../data'
    model_dir = '../model/weights'
    max_text_len = 400
    max_summary_len = 100
    embedding_dim = 256
    latent_dim = 512
    batch_size = 32
    epochs = 10
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading data...")
    texts, summaries = load_data(data_dir)
    
    # Split into train and validation sets
    train_texts, val_texts, train_summaries, val_summaries = train_test_split(
        texts, summaries, test_size=0.2, random_state=42
    )
    
    # Initialize data processor
    print("Preprocessing data...")
    processor = DataProcessor(max_text_len, max_summary_len)
    vocab_size = processor.fit_tokenizer(train_texts, train_summaries)
    
    # Prepare sequences
    train_encoder_input, train_decoder_input, train_decoder_target = processor.prepare_sequences(
        train_texts, train_summaries
    )
    val_encoder_input, val_decoder_input, val_decoder_target = processor.prepare_sequences(
        val_texts, val_summaries
    )
    
    # Initialize and build model
    print("Building model...")
    model = Seq2SeqSummarizer(vocab_size, embedding_dim, latent_dim)
    model.build_model()
    
    # Train model
    print("Training model...")
    history = model.train(
        train_encoder_input,
        train_decoder_input,
        train_decoder_target,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save model weights
    model.save_model(os.path.join(model_dir, 'seq2seq_weights.h5'))
    
    # Save vocabulary mappings
    vocab_data = {
        'word_to_index': processor.word_to_index,
        'index_to_word': processor.index_to_word
    }
    with open(os.path.join(model_dir, 'vocabulary.json'), 'w') as f:
        json.dump(vocab_data, f)
    
    print("Training completed!")

if __name__ == '__main__':
    main() 