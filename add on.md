CODE:

import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras import Model
import numpy as np

# ---- 1. Data / Preprocessing ----

# Your single parallel sentence
eng_sentences = ["He is reading a book"]
hin_sentences = ["वह एक पुस्तक पढ़ रहा है"]

# Special tokens
START_TOKEN = "<start>"
END_TOKEN = "<end>"

# Build vocabularies (word-level for simplicity)
eng_vocab = set()
hin_vocab = set()
for s in eng_sentences:
    for w in s.strip().split():
        eng_vocab.add(w)
for s in hin_sentences:
    for w in s.strip().split():
        hin_vocab.add(w)
# add special tokens to Hindi vocab
hin_vocab = hin_vocab.union({START_TOKEN, END_TOKEN})

eng_vocab = sorted(eng_vocab)
hin_vocab = sorted(hin_vocab)

eng2idx = {w: i+1 for i, w in enumerate(eng_vocab)}  # reserve 0 for padding
hin2idx = {w: i+1 for i, w in enumerate(hin_vocab)}

idx2eng = {i: w for w, i in eng2idx.items()}
idx2hin = {i: w for w, i in hin2idx.items()}

vocab_inp_size = len(eng2idx) + 1
vocab_tar_size = len(hin2idx) + 1

print("ENG vocab:", eng2idx)
print("HIN vocab:", hin2idx)

# Prepare sequences with START / END
eng_seqs = []
hin_in_seqs = []
hin_out_seqs = []
for eng, hin in zip(eng_sentences, hin_sentences):
    eng_ids = [eng2idx[w] for w in eng.strip().split()]
    hin_ids = [hin2idx[w] for w in hin.strip().split()]
    # decoder input: <start> + target words
    hin_in = [hin2idx[START_TOKEN]] + hin_ids
    # decoder target: target words + <end>
    hin_out = hin_ids + [hin2idx[END_TOKEN]]
    eng_seqs.append(eng_ids)
    hin_in_seqs.append(hin_in)
    hin_out_seqs.append(hin_out)

# Pad sequences
max_eng_len = max(len(s) for s in eng_seqs)
max_hin_len = max(len(s) for s in hin_in_seqs)

def pad_seq(seq, maxlen):
    return seq + [0] * (maxlen - len(seq))

encoder_input = np.array([pad_seq(s, max_eng_len) for s in eng_seqs])
decoder_input = np.array([pad_seq(s, max_hin_len) for s in hin_in_seqs])
decoder_target = np.array([pad_seq(s, max_hin_len) for s in hin_out_seqs])

print("encoder_input:", encoder_input)
print("decoder_input:", decoder_input)
print("decoder_target:", decoder_target)

# ---- 2. Define Models: Encoder, Attention, Decoder ----

class Encoder(Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(Encoder, self).__init__()
        self.units = units
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.lstm = LSTM(units, return_sequences=True, return_state=True)
    def call(self, x):
        x = self.embedding(x)
        enc_output, enc_h, enc_c = self.lstm(x)
        return enc_output, enc_h, enc_c

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
    def call(self, dec_hidden, enc_output):
        # dec_hidden: (batch, hidden)
        # enc_output: (batch, seq_len, hidden)
        dec_hidden_time = tf.expand_dims(dec_hidden, 1)  # (batch, 1, hidden)
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(dec_hidden_time)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch, hidden)
        return context_vector, attention_weights

class Decoder(Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(Decoder, self).__init__()
        self.units = units
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.lstm = LSTM(units, return_sequences=True, return_state=True)
        self.attention = BahdanauAttention(units)
        self.fc = Dense(vocab_size)
    def call(self, x, dec_hidden, enc_output):
        # x: (batch, seq_len) or (batch, 1) during inference
        context_vector, attn_weights = self.attention(dec_hidden, enc_output)
        x_emb = self.embedding(x)  # (batch, seq_len, embedding_dim)

        # Expand context_vector in time dimension and tile it
        context_vector = tf.expand_dims(context_vector, 1)  # (batch, 1, hidden)
        seq_len = tf.shape(x_emb)[1]
        context_vector = tf.tile(context_vector, [1, seq_len, 1])  # (batch, seq_len, hidden)

        # Concatenate along last axis
        x_concat = tf.concat([context_vector, x_emb], axis=-1)  # (batch, seq_len, hidden + embedding_dim)

        # Pass through LSTM
        output, state_h, state_c = self.lstm(x_concat)
        # Flatten time dimension for Dense
        output_flat = tf.reshape(output, (-1, output.shape[2]))
        logits = self.fc(output_flat)
        logits = tf.reshape(logits, (x.shape[0], seq_len, -1))
        return logits, state_h, state_c, attn_weights

# Hyperparameters
embedding_dim = 64
units = 128

encoder = Encoder(vocab_inp_size, embedding_dim, units)
decoder = Decoder(vocab_tar_size, embedding_dim, units)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none'
)

def loss_function(real, pred):
    # real: (batch, seq_len)
    # pred: (batch, seq_len, vocab)
    mask = tf.cast(tf.not_equal(real, 0), tf.float32)
    loss_ = loss_object(real, pred)
    loss_ *= mask
    # average over non-padding tokens
    return tf.reduce_sum(loss_) / (tf.reduce_sum(mask) + 1e-9)

@tf.function
def train_step(inp, targ, targ_inp):
    with tf.GradientTape() as tape:
        enc_output, enc_h, enc_c = encoder(inp)
        dec_hidden = enc_h
        predictions, _, _, _ = decoder(targ_inp, dec_hidden, enc_output)
        loss = loss_function(targ, predictions)
    vars = encoder.trainable_variables + decoder.trainable_variables
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))
    return loss

# ---- 3. Training ----

EPOCHS = 500  # adjust as needed
for epoch in range(EPOCHS):
    loss = train_step(encoder_input, decoder_target, decoder_input)
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.numpy():.4f}")

# ---- 4. Inference / Translation ----

def translate_sentence(inp_sentence):
    inp_ids = [eng2idx[w] for w in inp_sentence.strip().split()]
    inp_ids = pad_seq(inp_ids, max_eng_len)
    inp_array = np.array([inp_ids])
    enc_output, enc_h, enc_c = encoder(inp_array)
    dec_hidden = enc_h
    dec_input = np.array([[hin2idx[START_TOKEN]]])
    result_ids = []
    attention_plot = []

    for t in range(max_hin_len + 5):
        predictions, dec_h, dec_c, attn_w = decoder(dec_input, dec_hidden, enc_output)
        pred_id = tf.argmax(predictions[:, -1, :], axis=-1).numpy()[0]
        result_ids.append(pred_id)
        attention_plot.append(attn_w.numpy()[0])
        if pred_id == hin2idx[END_TOKEN]:
            break
        dec_input = np.array([[pred_id]])
        dec_hidden = dec_h

    result_words = []
    for rid in result_ids:
        if rid in idx2hin:
            w = idx2hin[rid]
            if w == END_TOKEN:
                break
            result_words.append(w)
    return result_words, attention_plot

# Test on your example
translated, attn = translate_sentence("He is reading a book")
print("Input:", "He is reading a book")
print("Predicted Hindi:", " ".join(translated))
print("Attention weights per time step:", attn)
OUTPUT:
<img width="504" height="483" alt="Screenshot 2025-10-08 120009" src="https://github.com/user-attachments/assets/6c8b0464-a185-41ca-86ae-492d8a8c1e2d" />
<img width="673" height="519" alt="Screenshot 2025-10-08 115941" src="https://github.com/user-attachments/assets/e70669f4-4fdc-4a6a-b598-6095e66f857d" />
