import streamlit as st
import joblib
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎬")

st.title("🎬 IMDB Movie Review Analyzer")
st.markdown("""
This app uses a **Machine Learning Pipeline** (TF-IDF + Logistic Regression) 
to classify movie reviews as either **Positive** or **Negative**.
""")

# --- 1. SETUP & INITIALIZATION ---
# These must run every time the app starts to ensure NLTK resources exist
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')

download_nltk_data()

# Global variables must match the names used in your training script
my_stemmer = PorterStemmer()
my_stop_words = set(stopwords.words('english'))

# --- 2. DEFINE PREPROCESSING FUNCTION ---
# This function must be identical to the one used during training
def preprocess_text(text_list):
    clean_data = []
    for text in text_list:
        # Lowercase & Normalization
        text = text.lower()
        text = ' '.join(text.split())
        
        # Remove URLs, Handles, Special Characters
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'\"]+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Stem & Remove Stopwords
        cleaned_tokens = [
            my_stemmer.stem(word) for word in tokens 
            if word not in my_stop_words and word not in string.punctuation
        ]
        clean_data.append(' '.join(cleaned_tokens))
    
    return clean_data

# --- 3. LOAD THE MODEL ---
@st.cache_resource
def load_model():
    # This loads the full pipeline (Preprocessing -> Vectorizer -> Model)
    pipeline = joblib.load('sentiment_pipeline.joblib')
    return pipeline

try:
    pipeline = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False



# Input Area
user_input = st.text_area("✍️ Enter a movie review:", height=150, placeholder="e.g., The movie was absolutely fantastic! The acting was great.")

# Prediction Logic
if st.button("Analyze Sentiment"):
    if not model_loaded:
        st.error("Model not loaded. Please check if 'sentiment_pipeline.joblib' exists.")
    elif user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner('Analyzing...'):
            # The pipeline handles raw text automatically!
            # Note: We pass [user_input] as a list because our function expects a list
            prediction = pipeline.predict([user_input])[0]
            probability = pipeline.predict_proba([user_input]).max()
            
            # Display Results
            st.divider()
            if prediction == 1: # Assuming 1 is Positive
                st.success(f"### 🎉 Result: POSITIVE")
                st.metric("Confidence Score", f"{probability:.2%}")
            else:
                st.error(f"### 👎 Result: NEGATIVE")
                st.metric("Confidence Score", f"{probability:.2%}")

# Footer
st.markdown("---")
st.caption("Built with Streamlit & Scikit-Learn")