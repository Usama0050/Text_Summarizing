import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Set page config FIRST
st.set_page_config(page_title="T5 Summarizer", layout="wide")

@st.cache_resource
def load_model():
    model_name = "Usamahf0050/t5_model"  # Hugging Face model path
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# UI
st.title("Text Summarizer")
text_input = st.text_area("Enter article or paragraph:", height=200)

if st.button("Summarize"):
    if text_input.strip() == "":
        st.warning("Please enter some text to summarize.")
    else:
        input_ids = tokenizer("summarize: " + text_input, return_tensors="pt", max_length=512, truncation=True).input_ids
        output_ids = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        st.subheader("Summary:")
        st.write(summary)
