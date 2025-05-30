import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Load local model and tokenizer
MODEL_PATH = "Usamahf0050/Bart_model"
tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)

# Title
st.title("Text Summarizer")
st.write("Summarize long text using Facebook's BART model.")

# Input text
input_text = st.text_area("Enter text to summarize:", height=300)

# Summarize button
if st.button("Summarize"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating summary..."):
            # Tokenize
            inputs = tokenizer([input_text], max_length=1024, return_tensors="pt", truncation=True)
            # Generate summary
            summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=50, max_length=200, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            # Display summary
            st.subheader("Summary:")
            st.success(summary)
