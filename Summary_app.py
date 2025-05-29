import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

@st.cache_resource
def load_model():
    model_name = "Usamahf0050/t5_model"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# UI
st.set_page_config(page_title="T5 Summarizer", layout="wide")
st.title("Text Summarizer")

text_input = st.text_area("Enter article or paragraph:", height=200)

if st.button("Summarize"):
    if text_input.strip():
        input_ids = tokenizer.encode("summarize: " + text_input, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(
            input_ids,
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.success("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")
