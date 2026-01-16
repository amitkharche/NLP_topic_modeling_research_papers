
---

# Topic Modeling for Research Papers

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange)

## Business Use Case

Understanding research trends is crucial for driving innovation, allocating funding, and guiding academic and industrial R&D. This project leverages **topic modeling** to extract hidden themes from a large corpus of research paper abstracts, allowing strategic insights into evolving fields and dominant ideas.

## Features Used

- `CountVectorizer` for converting abstract text into word-count vectors  
- `Latent Dirichlet Allocation (LDA)` for topic discovery  
- `Streamlit` interface to upload new data and view topic assignments  
- Top keywords extracted for each topic for better interpretability  
- CSV output with topic proportions for every abstract  
- Average topic distribution visualization  

## Pipeline Overview

### `model_training.py`
- Loads the dataset from `data/simulated_research_papers.csv`
- Preprocesses the abstract column using `CountVectorizer`
- Trains an LDA model with `n_components=5` using scikit-learn
- Extracts top 10 keywords per topic
- Saves the LDA model, vectorizer, and topic keyword list as `.pkl` files

### `app.py`
- Streamlit UI to upload a CSV file with an `Abstract` column
- Loads the pre-trained LDA model and vectorizer
- Assigns topic distributions to new abstracts
- Displays topic proportions along with readable topic names
- Visualizes average topic distribution using a bar chart
- Allows download of results as CSV

## How to Use

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Train the topic model
python model_training.py

# Step 3: Run the Streamlit web app
streamlit run app.py
````

## Project Structure

```
topic_modeling_research_papers/
├── data/
│   └── simulated_research_papers.csv    # Input dataset
├── model/
│   ├── lda_model.pkl                    # Trained LDA model
│   ├── vectorizer.pkl                   # CountVectorizer instance
│   └── topic_keywords.pkl               # List of top words for each topic
├── model_training.py                    # Script for model training and saving
├── app.py                               # Streamlit app for topic inference
├── requirements.txt                     # Required Python packages
└── README.md                            # Project documentation
```

## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute it.

---

## Contact

If you have questions or want to collaborate, feel free to connect with me on:

* [LinkedIn](https://www.linkedin.com/in/amitkharche)
* [Medium](https://medium.com/@amitkharche)
* [GitHub](https://github.com/amitkharche)
---
