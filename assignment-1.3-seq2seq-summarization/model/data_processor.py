import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re

class DataProcessor:
    def __init__(self, max_text_len=400, max_summary_len=100):
        self.max_text_len = max_text_len
        self.max_summary_len = max_summary_len
        self.tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        self.word_to_index = None
        self.index_to_word = None
        
    def preprocess_text(self, text):
        """Clean and preprocess text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def fit_tokenizer(self, texts, summaries):
        """Fit tokenizer on texts and summaries."""
        # Combine texts and summaries for vocabulary
        all_texts = texts + summaries
        
        # Fit tokenizer
        self.tokenizer.fit_on_texts(all_texts)
        
        # Create word-to-index and index-to-word mappings
        self.word_to_index = self.tokenizer.word_index
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}
        
        # Add special tokens
        self.word_to_index['<start>'] = len(self.word_to_index) + 1
        self.word_to_index['<end>'] = len(self.word_to_index) + 1
        self.index_to_word[len(self.word_to_index)] = '<start>'
        self.index_to_word[len(self.word_to_index)] = '<end>'
        
        return len(self.word_to_index)
    
    def prepare_sequences(self, texts, summaries):
        """Prepare input and target sequences for training."""
        # Convert texts to sequences
        text_sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Convert summaries to sequences and add start/end tokens
        summary_sequences = []
        for summary in summaries:
            seq = self.tokenizer.texts_to_sequences([summary])[0]
            seq = [self.word_to_index['<start>']] + seq + [self.word_to_index['<end>']]
            summary_sequences.append(seq)
        
        # Pad sequences
        encoder_input_data = pad_sequences(
            text_sequences,
            maxlen=self.max_text_len,
            padding='post'
        )
        
        decoder_input_data = pad_sequences(
            summary_sequences,
            maxlen=self.max_summary_len,
            padding='post'
        )
        
        # Create decoder target data (shifted by one)
        decoder_target_data = np.zeros_like(decoder_input_data)
        decoder_target_data[:, :-1] = decoder_input_data[:, 1:]
        
        return encoder_input_data, decoder_input_data, decoder_target_data
    
    def prepare_inference_input(self, text):
        """Prepare input sequence for inference."""
        # Preprocess text
        text = self.preprocess_text(text)
        
        # Convert to sequence
        sequence = self.tokenizer.texts_to_sequences([text])[0]
        
        # Pad sequence
        sequence = pad_sequences(
            [sequence],
            maxlen=self.max_text_len,
            padding='post'
        )
        
        return sequence
    
    def decode_sequence(self, sequence):
        """Convert sequence of indices back to text."""
        return ' '.join([self.index_to_word.get(idx, '') for idx in sequence if idx > 0])
    
    def split_into_sentences(self, text):
        """Split text into sentences."""
        return sent_tokenize(text)
    
    def get_word_count(self, text):
        """Get word count of text."""
        return len(word_tokenize(text))
    
    def get_sentence_count(self, text):
        """Get sentence count of text."""
        return len(sent_tokenize(text))
    
    def get_vocabulary_size(self):
        """Get size of vocabulary."""
        return len(self.word_to_index) 