"""
Streamlit app for topic modeling of research paper abstracts using LDA.
"""

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit UI setup
st.set_page_config(page_title="📚 Topic Modeling App", layout="wide")
st.title("📚 Topic Modeling for Research Papers")

# Load model, vectorizer, and topic keywords
@st.cache_resource
def load_model():
    try:
        with open("model/lda_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("model/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("model/topic_keywords.pkl", "rb") as f:
            topic_keywords = pickle.load(f)
        return model, vectorizer, topic_keywords
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, None

# Get topic distribution for new text data
def get_topic_distributions(lda, vectorizer, texts):
    X_counts = vectorizer.transform(texts)
    topic_distributions = lda.transform(X_counts)
    return topic_distributions

<<<<<<< HEAD
# Main app logic
=======
def display_topic_keywords(model, vectorizer, n_top_words=10):
    topic_keywords = {}
    for idx, topic in enumerate(model.components_):
        top_features = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topic_keywords[f"Topic_{idx+1}"] = top_features
    return topic_keywords

def plot_topic_distribution(df_topics):
    topic_sums = df_topics.sum().sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=topic_sums.index, y=topic_sums.values, ax=ax)
    ax.set_title("Overall Topic Distribution")
    ax.set_ylabel("Total Contribution")
    ax.set_xlabel("Topics")
    st.pyplot(fig)

def plot_pca(df_topics):
    pca = PCA(n_components=2)
    components = pca.fit_transform(df_topics)
    df_pca = pd.DataFrame(components, columns=["PC1", "PC2"])
    df_pca["Dominant_Topic"] = df_topics.idxmax(axis=1)
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Dominant_Topic", palette="tab10", ax=ax)
    ax.set_title("PCA of Topic Distributions")
    st.pyplot(fig)

>>>>>>> 01b84c966f14534bae05aa16814c2fdafec38d1e
def main():
    uploaded_file = st.file_uploader("📤 Upload a CSV file with an 'Abstract' column", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "Abstract" not in df.columns:
            st.error("❌ The uploaded CSV must contain an 'Abstract' column.")
            return

        st.subheader("📄 Preview of Uploaded Data")
        st.dataframe(df.head())

        model, vectorizer, topic_keywords = load_model()

        if model and vectorizer and topic_keywords:
            st.success("✅ Model loaded successfully!")

            topic_distributions = get_topic_distributions(model, vectorizer, df["Abstract"])
<<<<<<< HEAD
            topic_names = [name.split(":")[1] for name in topic_keywords]  # Extract readable names
            df_topics = pd.DataFrame(topic_distributions, columns=topic_names)
            df = pd.concat([df, df_topics], axis=1)

            st.subheader("🧠 Topic Distributions with Keywords")
            st.dataframe(df.head())

            st.subheader("📥 Download Results")
=======
            df_topics = pd.DataFrame(topic_distributions, columns=[f"Topic_{i+1}" for i in range(topic_distributions.shape[1])])
            df["Dominant_Topic"] = df_topics.idxmax(axis=1)
            df = pd.concat([df, df_topics], axis=1)

            st.subheader("Top Keywords per Topic")
            keywords = display_topic_keywords(model, vectorizer)
            for topic, words in keywords.items():
                st.markdown(f"**{topic}:** {', '.join(words)}")

            st.subheader("Filter by Dominant Topic")
            selected_topic = st.selectbox("Choose a topic", options=["All"] + sorted(df["Dominant_Topic"].unique()))
            if selected_topic != "All":
                df = df[df["Dominant_Topic"] == selected_topic]

            st.subheader("Topic Distributions")
            st.dataframe(df.head())

            st.subheader("Visualizations")
            plot_topic_distribution(df[[col for col in df.columns if col.startswith("Topic_")]])
            plot_pca(df[[col for col in df.columns if col.startswith("Topic_")]])

>>>>>>> 01b84c966f14534bae05aa16814c2fdafec38d1e
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Topic Modeling Output", csv, "topic_modeling_output.csv", "text/csv")

            with st.expander("🔍 Topic Keywords Used"):
                for keyword in topic_keywords:
                    st.markdown(f"- **{keyword}**")

            with st.expander("📊 Topic Distribution Plot (Avg across Abstracts)"):
                avg_dist = df_topics.mean().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(x=avg_dist.values, y=avg_dist.index, ax=ax)
                ax.set_title("Average Topic Distribution Across Abstracts")
                ax.set_xlabel("Average Proportion")
                st.pyplot(fig)

if __name__ == "__main__":
    main()
