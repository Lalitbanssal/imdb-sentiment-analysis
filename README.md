# 🎬 IMDB Movie Review Sentiment Analysis

## 📌 Project Overview
This is an end-to-end Machine Learning project that classifies movie reviews as **Positive** or **Negative**. 
It uses a **Logistic Regression** model trained on TF-IDF features (Unigrams + Bigrams), achieving **~90% accuracy** on the IMDB dataset.

## 🔧 Tech Stack
- **Python** (Pandas, NumPy)
- **Scikit-Learn** (Model Training & Pipeline)
- **NLTK** (Text Preprocessing: Tokenization & Stemming)
- **Streamlit** (Web Interface)
- **Docker** (Containerization)

## 🧠 Key Learnings
- Compared **Naive Bayes** vs. **Logistic Regression** (LogReg won).
- Experimented with **Text Preprocessing** (Stemming vs. Lemmatization).
- Found that including **Bigrams** (2-word context) significantly improved accuracy by capturing phrases like "not good".

## 🚀 How to Run Locally
1. Clone the repo:
   ```bash
   git clone [https://github.com/Lalitbanssal/imdb-sentiment-analysis.git](https://github.com/Lalitbanssal/imdb-sentiment-analysis.git)
2. Install dependencies:
    pip install -r requirements.txt
3. Run the app:
   streamlit run app.py
