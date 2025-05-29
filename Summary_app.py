import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load saved model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("t5_model")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# Web UI
st.title("📄 Article Summarizer using T5-small")
st.write("Enter your article or paragraph below and click 'Summarize'.")

text = st.text_area("Enter Text:", height=300)

if st.button("Summarize"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        input_text = "summarize: " + text.strip()
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

        summary_ids = model.generate(input_ids, max_length=60, min_length=15, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.subheader("📝 Summary")
        st.success(summary)