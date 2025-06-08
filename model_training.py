"""
Model training script for topic modeling using Latent Dirichlet Allocation (LDA).
"""

import os
import logging
import pandas as pd
import pickle
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "data/simulated_research_papers.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "lda_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

class TextSelector(BaseEstimator, TransformerMixin):
    """Select a single column from the DataFrame to use as text."""
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Dataset not found.")
    df = pd.read_csv(path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    return df

def train_model(df):
    count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    lda = LatentDirichletAllocation(n_components=5, max_iter=10, learning_method='online', random_state=42)

    logger.info("Fitting the LDA model...")
    X_counts = count_vectorizer.fit_transform(df['Abstract'])
    lda.fit(X_counts)

    return lda, count_vectorizer

def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved object to {path}")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data(DATA_PATH)
    lda_model, vectorizer = train_model(df)
    save_model(lda_model, MODEL_PATH)
    save_model(vectorizer, VECTORIZER_PATH)
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
