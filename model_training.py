"""
Model training script for topic modeling using Latent Dirichlet Allocation (LDA).
"""

import os
import logging
import pandas as pd
import pickle
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = "data/simulated_research_papers.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "lda_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
TOPIC_KEYWORDS_PATH = os.path.join(MODEL_DIR, "topic_keywords.pkl")

# Load dataset
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Dataset not found.")
    df = pd.read_csv(path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    return df

# Train LDA model
def train_model(df, n_topics=5):
    count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=10,
        learning_method='online',
        random_state=42
    )

    logger.info("Fitting the LDA model...")
    X_counts = count_vectorizer.fit_transform(df['Abstract'])
    lda.fit(X_counts)

    return lda, count_vectorizer

# Extract topic keywords
def get_topic_keywords(model, vectorizer, top_n=10):
    keywords = []
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-top_n - 1:-1]
        top_words = [feature_names[i] for i in top_features_ind]
        topic_keywords = " / ".join(top_words)
        keywords.append(f"Topic {topic_idx+1}: {topic_keywords}")
    return keywords

# Save model or any object
def save_model(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved object to {path}")

# Main function
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load and train
    df = load_data(DATA_PATH)
    lda_model, vectorizer = train_model(df)
    
    # Get keywords for each topic
    topic_keywords = get_topic_keywords(lda_model, vectorizer)

    # Save model components
    save_model(lda_model, MODEL_PATH)
    save_model(vectorizer, VECTORIZER_PATH)
    save_model(topic_keywords, TOPIC_KEYWORDS_PATH)

    logger.info("Training complete. LDA model, vectorizer, and topic keywords saved.")

if __name__ == "__main__":
    main()
