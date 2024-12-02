
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import tensorflow as tf

# Load the saved models and tokenizers
encoder_model = load_model("S:/new machine learning/encoder_model.keras")
decoder_model = load_model("S:/new machine learning/decoder_model.keras")
tokenizer_article = pickle.load(open("S:/new machine learning/x_tokenizer.pkl", 'rb'))
tokenizer_summary = pickle.load(open("S:/new machine learning/y_tokenizer.pkl", 'rb'))
reverse_target_word_index = tokenizer_summary.index_word
target_word_index = tokenizer_summary.word_index

# Function to preprocess the input text (article)
def preprocess_input_text(text, max_input_len):
    # Tokenizing the input text
    sequence = tokenizer_article.texts_to_sequences([text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_input_len, padding='post')
    return padded_sequence

# Decoding function to generate the summary from the model
def decode_sequence(input_seq, max_input_len, max_summary_len):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if sampled_token != 'eostok':
            decoded_sentence += ' ' + sampled_token

        # Exit condition: hit max length or find stop word.
        if sampled_token == 'eostok' or len(decoded_sentence.split()) >= max_summary_len:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

# Streamlit UI
def main():
    st.title("Text Summarization using Seq2Seq Model")

    st.write("""
    This app uses a sequence-to-sequence model to generate summaries for unseen articles. Simply paste an article and click "Generate Summary".
    """)

    # Input field for the article
    article = st.text_area("Enter Article", height=300)

    # Predefined constants (should match the training settings)
    max_input_len = 100  # Match the length used during training
    max_summary_len = 100

    if st.button("Generate Summary"):

        if article:
            # Preprocess the input article
            input_seq = preprocess_input_text(article, max_input_len)

            # Generate summary
            summary = decode_sequence(input_seq, max_input_len, max_summary_len)

            st.subheader("Generated Summary:")
            st.write(summary)

        else:
            st.error("Please enter an article.")

if __name__ == "__main__":
    main()
