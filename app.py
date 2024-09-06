import streamlit as st
import re
import pandas as pd
import requests
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib  
import os

ps = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Function to fetch news from News API
def fetch_news_from_newsapi():
    api_key = '1709fd72947744809e1cceb3f76d50d2'  
    url = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}'
    response = requests.get(url)
    articles = response.json().get('articles', [])
    return [article['title'] + ' ' + article['description'] for article in articles if article['description']]

def fetch_news_from_serpapi():
    api_key = '4bc65b1421a36fceee8af1d3ae8397726560ab9b1f7f0ccfbaf3357ef2b94d0b'  
    url = f'https://serpapi.com/search?api_key={api_key}&engine=google&q=python+programming'
    response = requests.get(url)
    articles = response.json().get('articles', [])
    return [article['title'] + ' ' + article['description'] for article in articles if article['description']]

news_data = fetch_news_from_newsapi() + fetch_news_from_serpapi()
news_df = pd.DataFrame(news_data, columns=['content'])
news_df['content'] = news_df['content'].apply(stemming)

news_df['label'] = [0 if i % 2 == 0 else 1 for i in range(len(news_df))] 

vector = TfidfVectorizer()
X = vector.fit_transform(news_df['content'])
y = news_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'logistic_model.pkl')
joblib.dump(vector, 'vectorizer.pkl')


# Add CSS styling
st.markdown("""
<style>
    .stApp {
        /*background-color: #add8e6;   Light gray background for the entire app */
        background-image: url('https://img.freepik.com/premium-photo/illustration-digital-network-around-world-background_1148129-22897.jpg?size=626&ext=jpg&ga=GA1.1.400743877.1725589726&semt=ais_hybrid');
            background-size: cover;  /* Make sure the image covers the entire background */
            background-position: right;  /* Center the background image */
            background-repeat: repeat;  /* Prevent the background image from repeating */
    }
    .reportview-container {
        background: transparent;
        padding: 20px;
    }
    
    .title {
        color: white;
        text-align: center;
        font-size: 50px;
        font-weight: bold;
        margin-top: 20px;
        font-family: sans serif;
    }
    
    .subheader {
        color: white;
        text-align: center;
        font-size: 24px;
        margin-bottom: 30px;
    }
    
    .analyze-btn {
        background-color: #ff9800;
        color: white;
        border: none;
        padding: 15px 32px;
        text-align: center;
        font-size: 16px;
        margin: 4px 2px;
        border-radius: 5px;
        cursor: pointer;
        width: 100%;
        transition: background-color 0.3s;
    }
    
    .analyze-btn:hover {
        background-color: #e68900;
    }
    
    .footer {
        font-size: 18px;
        text-align: center;
        padding: 10px;
        background-color: #1e3a8a;
        color: white;
        margin-top: 20px;
    }
    
    .result-box {
        padding: 20px;
        margin: 20px 0;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
    }
    
    .success-result {
        background-color: rgba(75, 170, 80, 0.7);
        color: white;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid rgba(75, 170, 80, 1);
        text-shadow: 
            -1px -1px 0 rgba(75, 170, 80, 1),
            1px -1px 0 rgba(75, 170, 80, 1),
            -1px 1px 0 rgba(75, 170, 80, 1),
            1px 1px 0 rgba(75, 170, 80, 1);
    }
    
    .error-result {
        background-color: rgba(170, 75, 80, 0.7);
        color: white;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid rgba(170, 75, 80, 1);
        text-shadow: 
            -1px -1px 0 rgba(170, 75, 80, 1),
            1px -1px 0 rgba(170, 75, 80, 1),
            -1px 1px 0 rgba(170, 75, 80, 1),
            1px 1px 0 rgba(170, 75, 80, 1);
    }
        .warning{
            font-size:20px}

        .text_area{
            font-size:20px}
            .sidebar .sidebar-content {
            background-color: #f0f2f6; /* Light background color */
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #4CAF50; /* Green border */
    }
    .sidebar h2 {
        color: white;
        font-size: 24px;a
        text-align: center;
        padding: 10px;
        border-radius: 10px;
        border: 2px solid white; /* Green border */
        margin-bottom: 40px;
        background-color: black;
    }
    .sidebar p {
        font-size: 16px;
        line-height: 1.5;
        text-align: justify;
        color: black;
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid black; 
    } 
    [data-testid="stSidebar"] {
        background-color: #808080;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
    <div class='sidebar'>
        <br><br>
        <h2>About</h2>
        <p>This is a simple Fake News Detection app built using Streamlit and Machine Learning. 
        It uses a Logistic Regression model to predict whether a news article is real or fake.
        The app retrieves the latest news from APIs and allows users to analyze the authenticity of a news article.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subheader'>Analyze whether a news article is Real or Fake</h3>", unsafe_allow_html=True)

input_text = st.text_area('📝 Enter the news article content:', height=150, placeholder="Paste the news article here...")
st.markdown("</div>", unsafe_allow_html=True)

# Prediction function
def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

# Button for analyzing news
if st.button('Analyze', help="Click to analyze if the news article is real or fake"):
    if input_text.strip():
        with st.spinner('Analyzing the news content...'):
            pred = prediction(input_text)
            if pred == 1:
                st.markdown("<div class='result-box error-result'>🚨 The News is Fake</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-box success-result'>✅ The News is Real</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠ Please enter some text for analysis")

# Footer
st.markdown("""
    <div class='footer'>
        Made with ❤ using Streamlit
    </div>
""", unsafe_allow_html=True)
