import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

class Seq2SeqSummarizer:
    def __init__(self, vocab_size, embedding_dim=256, latent_dim=512):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.model = None
        
    def build_model(self):
        # Encoder
        encoder_inputs = Input(shape=(None,))
        encoder_embedding = Embedding(self.vocab_size, self.embedding_dim)(encoder_inputs)
        encoder_lstm = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = Input(shape=(None,))
        decoder_embedding = Embedding(self.vocab_size, self.embedding_dim)(decoder_inputs)
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        decoder_dense = Dense(self.vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # Model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Encoder model for inference
        self.encoder = Model(encoder_inputs, encoder_states)
        
        # Decoder model for inference
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_embedding, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        
        self.decoder = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states
        )
        
    def train(self, encoder_input_data, decoder_input_data, decoder_target_data,
              batch_size=64, epochs=10, validation_split=0.2):
        """Train the model."""
        return self.model.fit(
            [encoder_input_data, decoder_input_data],
            decoder_target_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split
        )
    
    def predict(self, input_seq, max_summary_length, word_to_index, index_to_word):
        """Generate summary for input sequence."""
        # Encode the input sequence
        states_value = self.encoder.predict(input_seq)
        
        # Generate empty target sequence
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = word_to_index['<start>']
        
        # Initialize summary
        summary = []
        
        # Generate summary
        for _ in range(max_summary_length):
            output_tokens, h, c = self.decoder.predict(
                [target_seq] + states_value
            )
            
            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = index_to_word[sampled_token_index]
            
            # Exit if end token is predicted
            if sampled_word == '<end>':
                break
                
            summary.append(sampled_word)
            
            # Update target sequence and states
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]
        
        return ' '.join(summary)
    
    def save_model(self, filepath):
        """Save model weights."""
        self.model.save_weights(filepath)
    
    def load_model(self, filepath):
        """Load model weights."""
        self.model.load_weights(filepath) 