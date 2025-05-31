import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load local model
MODEL_PATH = "Usamahf0050/Bart_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

st.title("Text_Summarizer")
st.write("Summarize Your text.")

text = st.text_area("Enter your text here:", height=300)

if st.button("Summarize"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Summarizing..."):
            inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = model.generate(inputs["input_ids"], max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            st.subheader("Summary:")
            st.success(summary)
