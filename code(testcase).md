CODE:

import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras import Model
import numpy as np
# Sample vocab sizes
vocab_inp_size = 5000
vocab_tar_size = 5000
embedding_dim = 256
units = 512
# Encoder
class Encoder(Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.lstm = LSTM(self.enc_units, return_sequences=True, return_state=True)
    def call(self, x):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x)
        return output, state_h, state_c
# Bahdanau Attention
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
    def call(self, query, values):
        # query shape: (batch_size, hidden_size)
        # values shape: (batch_size, max_len, hidden_size)
        query_with_time_axis = tf.expand_dims(query, 1)  # (batch, 1, hidden)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch, max_len, 1)
        context_vector = attention_weights * values  # broadcasting
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch, hidden_size)
        return context_vector, attention_weights
# Decoder with attention
class Decoder(Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.lstm = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.fc = Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)
    def call(self, x, hidden, enc_output):
        # x: (batch, seq_len) during training (teacher forcing) or (batch, 1) during inference
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x_emb = self.embedding(x)  # (batch, seq_len, embedding_dim)
        # Expand context and tile it to match seq_len
        context_vector = tf.expand_dims(context_vector, 1)  # (batch, 1, hidden_size)
        seq_len = tf.shape(x_emb)[1]
        context_vector = tf.tile(context_vector, [1, seq_len, 1])  # (batch, seq_len, hidden_size)
        # Concatenate along last axis
        x_concat = tf.concat([context_vector, x_emb], axis=-1)  # (batch, seq_len, hidden_size + embedding_dim)
        # Pass through LSTM
        output, state_h, state_c = self.lstm(x_concat)
        # Flatten time dimension to feed fc
        output_flat = tf.reshape(output, (-1, output.shape[2]))  # (batch * seq_len, dec_units)
        logits = self.fc(output_flat)  # (batch * seq_len, vocab_size)
        # Reshape back5
        logits = tf.reshape(logits, (x.shape[0], seq_len, -1))  # (batch, seq_len, vocab_size)
        return logits, state_h, state_c, attention_weights
# Example usage / shape test
encoder = Encoder(vocab_inp_size, embedding_dim, units)
decoder = Decoder(vocab_tar_size, embedding_dim, units)
# Dummy inputs to test
batch_size = 2
inp_seq_len = 7
tar_seq_len = 5
dummy_enc_inp = tf.random.uniform((batch_size, inp_seq_len), maxval=vocab_inp_size, dtype=tf.int32)
enc_out, enc_h, enc_c = encoder(dummy_enc_inp)
dummy_dec_inp = tf.random.uniform((batch_size, tar_seq_len), maxval=vocab_tar_size, dtype=tf.int32)
dec_logits, dec_h, dec_c, attn = decoder(dummy_dec_inp, enc_h, enc_out)
print("enc_out shape:", enc_out.shape)        # should be (batch, inp_seq_len, units)
print("enc_h, enc_c shapes:", enc_h.shape, enc_c.shape)  # (batch, units)
print("dec_logits shape:", dec_logits.shape)  # (batch, tar_seq_len, vocab_tar_size)
print("attention shape:", attn.shape)         # (batch, inp_seq_len, 1)

OUTPUT:

<img width="310" height="75" alt="Screenshot 2025-10-08 120022" src="https://github.com/user-attachments/assets/b5100bf2-445d-4d66-88f7-9c1ae588d599" />
