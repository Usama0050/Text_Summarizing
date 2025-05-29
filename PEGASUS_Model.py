import streamlit as st
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch

@st.cache_resource
def load_model():
    model_path = "Usamahf0050/PEGASUS_Model"  # Path to local folder with saved PEGASUS model
    tokenizer = PegasusTokenizer.from_pretrained(model_path)
    model = PegasusForConditionalGeneration.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_model()

st.set_page_config(page_title="PEGASUS Summarizer", layout="wide")
st.title("PEGASUS Text Summarizer")

input_text = st.text_area("Enter the text to summarize", height=200)

if st.button("Summarize"):
    if input_text.strip():
        tokens = tokenizer(input_text, truncation=True, padding="longest", return_tensors="pt")
        summary_ids = model.generate(**tokens, max_length=100, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter text to summarize.")
