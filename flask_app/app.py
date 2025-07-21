import mlflow
from mlflow.tracking import MlflowClient
import joblib
from flask import Flask, request, jsonify, send_file
import os
import re
import string
import logging
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from io import BytesIO
import uuid
import random
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
from src.config.mlflow_config import setup_mlflow
setup_mlflow()

# -----------------------------------------------------------------------------
# Flask App Initialization
# -----------------------------------------------------------------------------
app = Flask(__name__)

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logger = logging.getLogger('api_logger')
logger.setLevel(logging.DEBUG)
os.makedirs('reports/logs', exist_ok=True)
file_handler = logging.FileHandler('reports/logs/api.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# -----------------------------------------------------------------------------
# NLTK Initialization
# -----------------------------------------------------------------------------
nltk.download("stopwords")
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()

# -----------------------------------------------------------------------------
# Preprocessing Config
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# load model and vectorizer
# -----------------------------------------------------------------------------

def load_model_and_vectorizer(model_name: str, version: int = None):
        client = MlflowClient()
        if version is None:
            # Load the latest version
            model_uri = f"models:/{model_name}/latest"
            # model_uri = f"models:/{model_name}/Production"
        else:
            model_uri = f"models:/{model_name}/{version}"
        
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Model {model_name} loaded successfully from version {version}.")

        vectorizer_path = os.path.join("models", "vectorizers", "tfidf3gram_vectorizer.pkl")
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer

model, vectorizer = load_model_and_vectorizer("lightGBM_model_v2", version=None)


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    """Clean and preprocess the input text."""
    text = text.strip().lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = text.replace('\n', ' ')  # Replace newlines with spaces
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove all non-ASCII characters
    text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)  # Remove non-alphabetic characters
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    words = [word for word in text.split() if word not in stop_words] # Remove stop words
    lemmatized = [lemmatizer.lemmatize(word) for word in words] # Lemmatize words
    cleaned_text = ' '.join(lemmatized) 
    return cleaned_text

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict sentiment of a comment."""
    data = request.get_json()
    if 'comments' not in data:
        return jsonify({'error': 'No comment provided'}), 400

    comments = data.get('comments')
    cleaned_comments = [preprocess_text(comment) for comment in comments]
    transformed_comments = vectorizer.transform(cleaned_comments)
    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        num_features = transformed_comments.shape[1]
        feature_names = [str(i) for i in range(num_features)]
    df_transformed_comments = pd.DataFrame(transformed_comments.toarray(), columns=feature_names)

    prediction = model.predict(df_transformed_comments).tolist()

    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, prediction)]
    return jsonify(response)

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    sentiment_counts = request.json.get("sentiment_counts", {})
    labels = ['Neutral', 'Positive', 'Negative']
    sizes = [
        sentiment_counts.get('0', 0),
        sentiment_counts.get('1', 0),
        sentiment_counts.get('2', 0)
    ]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        textprops={'fontsize': 14}
    )
    for autotext in autotexts:
        autotext.set_fontsize(12)

    ax.axis('equal')
    output = BytesIO()
    plt.tight_layout()
    plt.savefig(output, format='png', bbox_inches='tight')
    output.seek(0)
    return send_file(output, mimetype='image/png')

@app.route("/generate_trend_graph", methods=["POST"])
def generate_trend_graph():
    try:
        sentiment_data = request.json.get("sentiment_data", [])
        if not sentiment_data:
            raise ValueError("No sentiment data provided.")

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)

        df['month'] = df['timestamp'].dt.to_period('M').dt.to_timestamp()
        df['sentiment'] = df['sentiment'].map({0: "Neutral", 1: "Positive", 2: "Negative"})
        df['count'] = 1

        trend = df.groupby(['month', 'sentiment'])['count'].count().unstack().fillna(0)

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        trend.plot(ax=ax, marker='o', linewidth=2)

        ax.set_title("Monthly Sentiment Trend")
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Comments")
        ax.legend(title="Sentiment")
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        output = BytesIO()
        plt.savefig(output, format="png", bbox_inches="tight")
        output.seek(0)
        return send_file(output, mimetype="image/png")

    except Exception as e:
        logger.error(f"Error generating trend graph: {e}")
        return jsonify({"error": str(e)}), 500


GEMINI_API_KEY = "AIzaSyDStfTRZ2MuOXzH-00_21KegNppcMVmcJc"  # Replace with your key
# GEMINII_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
@app.route("/summarize_video", methods=["POST"])
def summarize_video():
    try:
        video_id = request.json.get("video_id")
        print("\n\n\nGEMINI_API_KEY: ", GEMINI_API_KEY, "\nvideo_id: ", video_id,"\n\n\n")
        if not video_id:
            return jsonify({"error": "video_id is required"}), 400
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        if not transcript:
            return jsonify({"error": "transcript not fetching"}), 400
        full_text = " ".join([line['text'] for line in transcript])
        prompt = f'Summarize this youtube video transcript and also give result with punctuations \ntext = "{full_text}."'

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=prompt
        )

        response = model.generate_content("now give summary of this video transcript")
        return jsonify({"summary": response.text})

    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return jsonify({"error": str(e)}), 500

def clean_text(text: str) -> str:
    try:
        text = text.strip().lower()
        text = url_pattern.sub('', text)
        text = text.replace('\n', ' ')
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        words = [word for word in text.split() if word not in stop_words]
        lemmatized = [lemmatizer.lemmatize(word) for word in words]
        cleaned_text = ' '.join(lemmatized)
        return cleaned_text
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return ""

@app.route("/health")
def health():
    return "OK", 200


@app.route('/generate_wordcloud', methods=['POST'])
def wordcloud():
    comments = request.json.get("comments", [])
    text = " ".join(clean_text(comment) for comment in comments)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    output = BytesIO()
    plt.savefig(output, format='png')
    output.seek(0)
    return send_file(output, mimetype='image/png')

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
