CODE:
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np

# Sample data
input_texts = ['I love NLP', 'He plays football']
target_texts = [['PRON', 'VERB', 'NOUN'], ['PRON', 'VERB', 'NOUN']]

# Build vocabularies
word_vocab = sorted(set(word for sent in input_texts for word in sent.split()))
tag_vocab = sorted(set(tag for tags in target_texts for tag in tags))

word2idx = {w: i+1 for i, w in enumerate(word_vocab)}  # reserve 0 for padding
tag2idx = {t: i+1 for i, t in enumerate(tag_vocab)}

num_encoder_tokens = len(word2idx) + 1
num_decoder_tokens = len(tag2idx) + 1

# Convert texts to sequences of indices
max_encoder_seq_length = max(len(sent.split()) for sent in input_texts)
max_decoder_seq_length = max(len(tags) for tags in target_texts)

encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length), dtype='int32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length), dtype='int32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (in_txt, tgt_tags) in enumerate(zip(input_texts, target_texts)):
    in_words = in_txt.split()
    for t, w in enumerate(in_words):
        encoder_input_data[i, t] = word2idx[w]
    for t, tag in enumerate(tgt_tags):
        decoder_input_data[i, t] = tag2idx[tag]
        # decoder target is one step ahead
        if t > 0:
            decoder_target_data[i, t-1, tag2idx[tag]] = 1.0
    # Also consider “end-of-decoder” – optionally you can add a special token

# Define model
latent_dim = 256

# Encoder
encoder_inputs = Input(shape=(None,), name='encoder_inputs')
enc_emb = Embedding(num_encoder_tokens, latent_dim, mask_zero=True, name='encoder_embedding')(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True, name='encoder_lstm')
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,), name='decoder_inputs')
dec_emb = Embedding(num_decoder_tokens, latent_dim, mask_zero=True, name='decoder_embedding')(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Train
model.fit([encoder_input_data, decoder_input_data],
          decoder_target_data,
          batch_size=2,
          epochs=100,
          verbose=2)

# --- Inference models ---
# Encoder inference model
encoder_model_inf = Model(encoder_inputs, encoder_states)

# Decoder inference model
decoder_state_input_h = Input(shape=(latent_dim,), name='input_h')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_c')
dec_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = decoder_embedding = model.get_layer('decoder_embedding')(decoder_inputs)
# (Note: better to re-use layer objects rather than re-calling new ones — here for clarity)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=dec_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model_inf = Model(
    [decoder_inputs] + dec_states_inputs,
    [decoder_outputs2] + decoder_states2
)

# Reverse index maps
idx2word = {i: w for w, i in word2idx.items()}
idx2tag = {i: t for t, i in tag2idx.items()}

def decode_sequence(input_seq):
    # Encode the input as state vectors
    states_value = encoder_model_inf.predict(input_seq)

    # Generate empty target sequence of length 1 (start token)
    target_seq = np.zeros((1, 1), dtype='int32')
    # We don’t have an explicit “start” token in this toy example,
    # so we can just leave target_seq[0,0] = 0 (padding index) or some default.

    stop_condition = False
    decoded_tags = []
    while not stop_condition:
        output_tokens, h, c = decoder_model_inf.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_tag = idx2tag.get(sampled_token_index, None)
        decoded_tags.append(sampled_tag)

        # Exit condition
        if sampled_tag is None or len(decoded_tags) >= max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (length 1)
        target_seq = np.zeros((1, 1), dtype='int32')
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_tags

# Test decode
for seq_index in range(len(input_texts)):
    input_seq = encoder_input_data[seq_index: seq_index+1]
    decoded = decode_sequence(input_seq)
    print('Input:', input_texts[seq_index])
    print('Decoded POS tags:', decoded)
    print('Ground truth:', target_texts[seq_index])
    OUTPUT:
    <img width="342" height="169" alt="Screenshot 2025-10-08 115848" src="https://github.com/user-attachments/assets/3e2c4ff1-cdce-4d94-b34b-86339f62a29b" />
