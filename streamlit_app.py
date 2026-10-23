import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.title("ABSA Demo")

@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained(".")
    model = AutoModelForSequenceClassification.from_pretrained(".")
    model.eval()
    return tok, model

tok, model = load_model()

text = st.text_area("Review text")
aspect = st.text_input("Aspect")

if st.button("Analyze") and text and aspect:
    inputs = tok(aspect, text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = logits.argmax(-1).item()
    st.write("Prediction:", model.config.id2label[pred])