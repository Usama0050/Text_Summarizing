import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.title("Text Summarizer")

# Load tokenizer and model from local directory
MODEL_PATH = "./bart_summarizer"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# Ensure model is on CPU
model.to("cpu")

# Text input
text = st.text_area("Enter your text to summarize:", height=300)

# When user clicks 'Summarize'
if st.button("Summarize"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating summary..."):
            inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
            input_ids = inputs["input_ids"].to("cpu")
            attention_mask = inputs["attention_mask"].to("cpu")

            summary_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=200,
                min_length=50,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            st.subheader("Summary:")
            st.success(summary)
