"""
Streamlit app for topic modeling of research paper abstracts.
"""

import streamlit as st
import pandas as pd
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="📚 Topic Modeling App", layout="wide")
st.title("📚 Topic Modeling for Research Papers")

@st.cache_resource
def load_model():
    try:
        with open("model/lda_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("model/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

def get_topic_distributions(lda, vectorizer, texts):
    X_counts = vectorizer.transform(texts)
    topic_distributions = lda.transform(X_counts)
    return topic_distributions

def main():
    uploaded_file = st.file_uploader("Upload CSV with 'Abstract' column", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "Abstract" not in df.columns:
            st.error("CSV must contain an 'Abstract' column.")
            return

        st.subheader("Uploaded Data")
        st.dataframe(df.head())

        model, vectorizer = load_model()
        if model and vectorizer:
            topic_distributions = get_topic_distributions(model, vectorizer, df["Abstract"])
            df_topics = pd.DataFrame(topic_distributions, columns=[f"Topic_{i+1}" for i in range(topic_distributions.shape[1])])
            df = pd.concat([df, df_topics], axis=1)

            st.subheader("Topic Distributions")
            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Topic Distributions as CSV", csv, "topic_modeling_output.csv", "text/csv")

if __name__ == "__main__":
    main()
