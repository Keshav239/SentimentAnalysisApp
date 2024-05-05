import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import numpy as np

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    return tokenizer, model

tokenizer, model = load_model()

st.title("Sentiment Analysis App")

text = st.text_input("Enter text to analyze:")
threshold = st.slider("Set sentiment strength threshold:", 0.0, 1.0, 0.5, 0.01)

if st.button("Analyze") and text:
    encoding = tokenizer.encode_plus(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    with torch.no_grad():
        output = model(input_ids, attention_mask)
    logits = output.logits.squeeze()

    num_classes = logits.shape[0]
    sentiments = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"][:num_classes]

    softmax = torch.nn.Softmax(dim=0)
    probabilities = softmax(logits).numpy()

    prediction = int(torch.argmax(logits))
    sentiment = sentiments[prediction]
    st.write(f"Detected Sentiment: {sentiment}")

    # Normalize scores for display
    values = probabilities.tolist()

    fig, ax = plt.subplots()
    colors = plt.cm.coolwarm(np.linspace(0, 1, num_classes))
    bars = ax.bar(sentiments, values, color=colors)

    # Highlight bars that pass the threshold
    for bar, value in zip(bars, values):
        if value > threshold:
            bar.set_alpha(1.0)  # Solid color for high confidence
        else:
            bar.set_alpha(0.5)  # Faded color for low confidence

    ax.set_title("Sentiment Analysis Scores with Confidence Threshold")
    ax.set_ylabel("Confidence")
    st.pyplot(fig)