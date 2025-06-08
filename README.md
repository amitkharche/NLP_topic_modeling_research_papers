# 📚 Topic Modeling for Research Papers

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange)

## 📌 Business Use Case

Understanding trends in academic research is essential for funding, innovation, and policy. Topic modeling helps uncover hidden themes in large volumes of research papers, enabling strategic decisions in R&D and education.

## 🧠 Features Used

- Abstract text from research papers
- CountVectorizer for bag-of-words representation
- Latent Dirichlet Allocation (LDA) for topic modeling

## ⚙️ Pipeline Overview

### `model_training.py`
- Load data
- Convert abstract text to count vectors
- Train LDA model
- Save LDA model and vectorizer

### `app.py`
- Upload abstracts via Streamlit
- Run model to assign topics
- Download topic distribution results

## 🚀 How to Use

```bash
pip install -r requirements.txt
python model_training.py
streamlit run app.py
```

## 🗂 Project Structure

```
topic_modeling_research_papers/
├── data/                         # CSV dataset of research abstracts
├── model/                        # LDA model and vectorizer
├── app.py                        # Streamlit app
├── model_training.py             # Training pipeline
├── requirements.txt              # Dependencies
├── README.md                     # Project documentation
└── .github/                      # Issue templates and CI
```

## 📄 License

This project is licensed under the MIT License.
