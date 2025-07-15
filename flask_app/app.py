from flask import Flask, request, jsonify
import flask_cors
from flask_cors import CORS
import os
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import joblib
import re
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words_to_keep = {
    'not', 'no', 'never', 'none', 'nothing', 'nowhere', 'neither', 'nor', 'cannot',
    "can't", "won't", "isn't", "aren't", "wasn't", "weren't", "don't", "didn't", "doesn't",
    "hasn't", "haven't", "hadn't", "shouldn't", "wouldn't", "couldn't", "mustn't",
    'but', 'however', 'yet', 'though', 'although', 'even though', 'still',
    'very', 'too', 'extremely', 'highly', 'so', 'such', 'absolutely', 'completely', 'totally', 'utterly', 'really',
    'slightly', 'somewhat', 'fairly', 'rather', 'little', 'bit', 'barely', 'hardly', 'scarcely'
}
stop_words = set(stopwords.words('english')) - words_to_keep
url_pattern = re.compile(r'https?://\S+|www\.\S+')

app = Flask(__name__)
# Allow CORS for Chrome Extensions and localhost
CORS(app, origins=[
    "chrome-extension://goligagcdhmagbapncddhglnbilikned",
    "http://localhost:5000",
    "http://127.0.0.1:5000"
])

def preprocess_text(text: str) -> str:
    """Clean and preprocess the input text."""
    text = text.strip().lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = text.replace('\n', ' ')  # Replace newlines with spaces
    text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)  # Remove non-alphabetic characters
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    words = [word for word in text.split() if word not in stop_words] # Remove stop words
    lemmatized = [lemmatizer.lemmatize(word) for word in words] # Lemmatize words
    cleaned_text = ' '.join(lemmatized) 
    return cleaned_text

def load_model_and_vectorizer(model_name: str, version: int = None):
        # mlflow.set_tracking_uri("http://")
        client = MlflowClient()
        if version is None:
            # Load the latest version
            model_uri = f"models:/{model_name}/latest"
        else:
            model_uri = f"models:/{model_name}/{version}"
        
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Model {model_name} loaded successfully from version {version}.")

        vectorizer_path = os.path.join("models", "vectorizers", "tfidf3gram_vectorizer.pkl")
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer

# def load_model_and_vectorizer():
#     """Load the model and vectorizer from MLflow Model Registry."""
#     model_path = os.path.join("models", "model", "lightGBM_model_v2.pkl")
#     model = joblib.load(model_path)
#     return model, vectorizer


model, vectorizer = load_model_and_vectorizer("lightGBM_model_v2", version=2)
from flask import send_file
import matplotlib.pyplot as plt
import io

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    data = request.get_json()
    sentiment_counts = data.get('sentiment_counts', {})
    labels = ['Positive', 'Neutral', 'Negative']
    values = [
        sentiment_counts.get("1", 0),
        sentiment_counts.get("0", 0),
        sentiment_counts.get("2", 0)
    ]
    colors = ["#1b5b1a", "#888888", "#b62c2c"]

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

from flask import request, send_file
import matplotlib.pyplot as plt
import io
import pandas as pd
from datetime import datetime

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    data = request.get_json()
    sentiment_data = data.get('sentiment_data', [])
    if not sentiment_data:
        return "No data", 400

    df = pd.DataFrame(sentiment_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])

    # Use only year and month for grouping
    df['month'] = df['timestamp'].dt.to_period('M').dt.to_timestamp()

    # Map sentiment values to int if needed
    df['sentiment'] = df['sentiment'].astype(int)

    # Count each sentiment per month
    pos = df[df['sentiment'] == 1].groupby('month').size()
    neu = df[df['sentiment'] == 0].groupby('month').size()
    neg = df[df['sentiment'].isin([-1, 2])].groupby('month').size()  # Use -1 or 2 for negative

    # Union of all months
    all_months = pd.date_range(df['month'].min(), df['month'].max(), freq='MS')
    pos = pos.reindex(all_months, fill_value=0)
    neu = neu.reindex(all_months, fill_value=0)
    neg = neg.reindex(all_months, fill_value=0)

    # Plot
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(pos.index, pos.values, color='#1b5b1a', marker='o', label='Positive')
    ax.plot(neu.index, neu.values, color='#444444', marker='o', label='Neutral')
    ax.plot(neg.index, neg.values, color='#b62c2c', marker='o', label='Negative')
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Comments')
    ax.set_title('Monthly Sentiment Trend')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

from wordcloud import WordCloud

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    data = request.get_json()
    comments = data.get('comments', [])
    text = ' '.join(comments)
    wc = WordCloud(width=400, height=200, background_color='white').generate(text)
    buf = io.BytesIO()
    wc.to_image().save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict sentiment of a comment."""
    data = request.get_json()
    if 'comments' not in data:
        return jsonify({'error': 'No comment provided'}), 400

    comments = data.get('comments')
    cleaned_comments = [preprocess_text(comment) for comment in comments]
    transformed_comments = vectorizer.transform(cleaned_comments)

    prediction = model.predict(transformed_comments).tolist()

    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, prediction)]
    return jsonify(response)


def clean_text(text: str) -> str:
        text = text.strip().lower()
        text = url_pattern.sub('', text)
        text = text.replace('\n', ' ')
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        words = [word for word in text.split() if word not in stop_words]
        lemmatized = [lemmatizer.lemmatize(word) for word in words]
        cleaned_text = ' '.join(lemmatized)
        return cleaned_text
        return ""


if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5000, debug=True)


