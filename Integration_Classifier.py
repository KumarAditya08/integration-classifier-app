# app.py

import streamlit as st
import pandas as pd
import re
from joblib import load
from sklearn.metrics import accuracy_score

@st.cache_resource
def load_everything():
    vec = load("vectorizer.pkl")
    mdl = load("model.pkl")
    df = pd.read_csv("dataset.csv")
    df["clean"] = df["question_text"].apply(lambda t: re.sub(r'[^a-z0-9 ]',' ', t.lower()))
    X = vec.transform(df["clean"])
    acc = accuracy_score(df["label"], mdl.predict(X))
    return vec, mdl, acc

vec, mdl, full_acc = load_everything()

st.title("🔍 Integration Method Classifier")
st.sidebar.header("Options")
if st.sidebar.button("Show Model Accuracy"):
    st.sidebar.success(f"Model Accuracy: {full_acc*100:.2f}% (on full dataset)")

st.markdown("Enter an integration question (e.g., `Integral of x*log(x)`):")
user_input = st.text_input("")

if st.button("Predict Method"):
    clean = re.sub(r'[^a-z0-9 ]',' ', user_input.lower())
    pred = mdl.predict(vec.transform([clean]))[0]
    st.success(f"✅ Predicted Method: **{pred.replace('_',' ').title()}**")
