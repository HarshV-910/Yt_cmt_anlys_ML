# import mlflow
# from mlflow.tracking import MlflowClient
# import joblib
# from flask import Flask, request, jsonify, send_file
# import os
# import re
# import string
# import logging
# import nltk
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from wordcloud import WordCloud
# from io import BytesIO
# import uuid
# import random
# from youtube_transcript_api import YouTubeTranscriptApi
# import google.generativeai as genai
# from src.config.mlflow_config import setup_mlflow
# setup_mlflow()

# # -----------------------------------------------------------------------------
# # Flask App Initialization
# # -----------------------------------------------------------------------------
# app = Flask(__name__)

# # -----------------------------------------------------------------------------
# # Logging Configuration
# # -----------------------------------------------------------------------------
# logger = logging.getLogger('api_logger')
# logger.setLevel(logging.DEBUG)
# os.makedirs('reports/logs', exist_ok=True)
# file_handler = logging.FileHandler('reports/logs/api.log')
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

# # -----------------------------------------------------------------------------
# # NLTK Initialization
# # -----------------------------------------------------------------------------
# nltk.download("stopwords")
# nltk.download("wordnet")
# lemmatizer = WordNetLemmatizer()

# # -----------------------------------------------------------------------------
# # Preprocessing Config
# # -----------------------------------------------------------------------------
# words_to_keep = {
#     'not', 'no', 'never', 'none', 'nothing', 'nowhere', 'neither', 'nor', 'cannot',
#     "can't", "won't", "isn't", "aren't", "wasn't", "weren't", "don't", "didn't", "doesn't",
#     "hasn't", "haven't", "hadn't", "shouldn't", "wouldn't", "couldn't", "mustn't",
#     'but', 'however', 'yet', 'though', 'although', 'even though', 'still',
#     'very', 'too', 'extremely', 'highly', 'so', 'such', 'absolutely', 'completely', 'totally', 'utterly', 'really',
#     'slightly', 'somewhat', 'fairly', 'rather', 'little', 'bit', 'barely', 'hardly', 'scarcely'
# }
# stop_words = set(stopwords.words('english')) - words_to_keep
# url_pattern = re.compile(r'https?://\S+|www\.\S+')

# # -----------------------------------------------------------------------------
# # load model and vectorizer
# # -----------------------------------------------------------------------------

# def load_model_and_vectorizer(model_name: str, version: int = None):
#         client = MlflowClient()
#         if version is None:
#             # Load the latest version
#             model_uri = f"models:/{model_name}/latest"
#             # model_uri = f"models:/{model_name}/Production"
#         else:
#             model_uri = f"models:/{model_name}/{version}"
        
#         model = mlflow.pyfunc.load_model(model_uri)
#         print(f"Model {model_name} loaded successfully from version {version}.")

#         vectorizer_path = os.path.join("models", "vectorizers", "tfidf3gram_vectorizer.pkl")
#         vectorizer = joblib.load(vectorizer_path)
#         return model, vectorizer

# model, vectorizer = load_model_and_vectorizer("lightGBM_model_v2", version=None)


# # -----------------------------------------------------------------------------
# # API Endpoints
# # -----------------------------------------------------------------------------
# def preprocess_text(text: str) -> str:
#     """Clean and preprocess the input text."""
#     text = text.strip().lower()
#     text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
#     text = text.replace('\n', ' ')  # Replace newlines with spaces
#     text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove all non-ASCII characters
#     text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)  # Remove non-alphabetic characters
#     text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
#     words = [word for word in text.split() if word not in stop_words] # Remove stop words
#     lemmatized = [lemmatizer.lemmatize(word) for word in words] # Lemmatize words
#     cleaned_text = ' '.join(lemmatized) 
#     return cleaned_text

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Endpoint to predict sentiment of a comment."""
#     data = request.get_json()
#     if 'comments' not in data:
#         return jsonify({'error': 'No comment provided'}), 400

#     comments = data.get('comments')
#     cleaned_comments = [preprocess_text(comment) for comment in comments]
#     transformed_comments = vectorizer.transform(cleaned_comments)
#     try:
#         feature_names = vectorizer.get_feature_names_out()
#     except AttributeError:
#         num_features = transformed_comments.shape[1]
#         feature_names = [str(i) for i in range(num_features)]
#     df_transformed_comments = pd.DataFrame(transformed_comments.toarray(), columns=feature_names)

#     prediction = model.predict(df_transformed_comments).tolist()

#     response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, prediction)]
#     return jsonify(response)

# @app.route('/generate_chart', methods=['POST'])
# def generate_chart():
#     sentiment_counts = request.json.get("sentiment_counts", {})
#     labels = ['Neutral', 'Positive', 'Negative']
#     sizes = [
#         sentiment_counts.get('0', 0),
#         sentiment_counts.get('1', 0),
#         sentiment_counts.get('2', 0)
#     ]

#     fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
#     wedges, texts, autotexts = ax.pie(
#         sizes,
#         labels=labels,
#         autopct='%1.1f%%',
#         startangle=140,
#         textprops={'fontsize': 14}
#     )
#     for autotext in autotexts:
#         autotext.set_fontsize(12)

#     ax.axis('equal')
#     output = BytesIO()
#     plt.tight_layout()
#     plt.savefig(output, format='png', bbox_inches='tight')
#     output.seek(0)
#     return send_file(output, mimetype='image/png')

# @app.route("/generate_trend_graph", methods=["POST"])
# def generate_trend_graph():
#     try:
#         sentiment_data = request.json.get("sentiment_data", [])
#         if not sentiment_data:
#             raise ValueError("No sentiment data provided.")

#         df = pd.DataFrame(sentiment_data)
#         df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
#         df.dropna(subset=['timestamp'], inplace=True)

#         df['month'] = df['timestamp'].dt.to_period('M').dt.to_timestamp()
#         df['sentiment'] = df['sentiment'].map({0: "Neutral", 1: "Positive", 2: "Negative"})
#         df['count'] = 1

#         trend = df.groupby(['month', 'sentiment'])['count'].count().unstack().fillna(0)

#         # Plotting
#         fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
#         trend.plot(ax=ax, marker='o', linewidth=2)

#         ax.set_title("Monthly Sentiment Trend")
#         ax.set_xlabel("Month")
#         ax.set_ylabel("Number of Comments")
#         ax.legend(title="Sentiment")
#         ax.grid(True)
#         plt.xticks(rotation=45)
#         plt.tight_layout()

#         output = BytesIO()
#         plt.savefig(output, format="png", bbox_inches="tight")
#         output.seek(0)
#         return send_file(output, mimetype="image/png")

#     except Exception as e:
#         logger.error(f"Error generating trend graph: {e}")
#         return jsonify({"error": str(e)}), 500


# # GENAI_API_KEY = "AIzaSyDStfTRZ2MuOXzH-00_21KegNppcMVmcJc"  # Replace with your key
# GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
# genai.configure(api_key=GENAI_API_KEY)
# @app.route("/summarize_video", methods=["POST"])
# def summarize_video():
#     try:
#         video_id = request.json.get("video_id")
#         transcript = YouTubeTranscriptApi.get_transcript(video_id)
#         full_text = " ".join([line['text'] for line in transcript])
#         prompt = f'Summarize this youtube video transcript and also give result with punctuations \ntext = "{full_text}."'

#         model = genai.GenerativeModel(
#             model_name="gemini-2.0-flash",
#             system_instruction=prompt
#         )

#         response = model.generate_content("now give summary of this video transcript")
#         return jsonify({"summary": response.text})

#     except Exception as e:
#         logger.error(f"Summary generation failed: {e}")
#         return jsonify({"error": str(e)}), 500

# def clean_text(text: str) -> str:
#     try:
#         text = text.strip().lower()
#         text = url_pattern.sub('', text)
#         text = text.replace('\n', ' ')
#         text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
#         words = [word for word in text.split() if word not in stop_words]
#         lemmatized = [lemmatizer.lemmatize(word) for word in words]
#         cleaned_text = ' '.join(lemmatized)
#         return cleaned_text
#     except Exception as e:
#         logger.error(f"Error processing text: {e}")
#         return ""

# @app.route('/generate_wordcloud', methods=['POST'])
# def wordcloud():
#     comments = request.json.get("comments", [])
#     text = " ".join(clean_text(comment) for comment in comments)
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
#     fig, ax = plt.subplots()
#     ax.imshow(wordcloud, interpolation='bilinear')
#     ax.axis('off')
#     output = BytesIO()
#     plt.savefig(output, format='png')
#     output.seek(0)
#     return send_file(output, mimetype='image/png')

# # -----------------------------------------------------------------------------
# # Run
# # -----------------------------------------------------------------------------
# if __name__ == '__main__':
#     app.run(debug=True)


# # -----------------------------------------------------------------------------
# # -----------------------------------------------------------------------------
# # -----------------------------------------------------------------------------
# # -----------------------------------------------------------------------------
# # -----------------------------------------------------------------------------

import mlflow
from mlflow.tracking import MlflowClient
import joblib
from flask import Flask, request, jsonify, send_file
import os
import re
import string
import logging
import sys # Keep this import for explicit stderr prints
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from io import BytesIO
import uuid
import random # Not used, can be removed if not planned for future use
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
import google.generativeai as genai

# Assuming src.config.mlflow_config exists and setup_mlflow is defined there
from src.config.mlflow_config import setup_mlflow

# -----------------------------------------------------------------------------
# Global Flask App Initialization (SINGLE DEFINITION)
# -----------------------------------------------------------------------------
app = Flask(__name__)

# -----------------------------------------------------------------------------
# Logging Configuration (SINGLE DEFINITION)
# This configures the root logger and ensures all messages go to stderr
# for Gunicorn to capture into flask_app.log via 2>&1 redirection.
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG, # Set to DEBUG to capture *everything* for detailed debugging
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr) # FORCE logs to stderr
    ]
)
# Get a logger instance for this module
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# NLTK Initialization
# -----------------------------------------------------------------------------
logger.info("Downloading NLTK data (stopwords, wordnet)...")
# Use quiet=True to suppress verbose download messages
try:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    lemmatizer = WordNetLemmatizer()
    logger.info("NLTK data downloaded and lemmatizer initialized.")
except Exception as e:
    logger.critical(f"FATAL: Failed to download NLTK data: {e}")
    print(f"FATAL ERROR: NLTK download failed: {e}", file=sys.stderr)
    sys.exit(1) # Crash if NLTK data is essential and fails to download

# -----------------------------------------------------------------------------
# Preprocessing Configuration
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
# MLflow Setup & Model Loading (SINGLE DEFINITION - happens once on app startup)
# -----------------------------------------------------------------------------
# Setup MLflow tracking
setup_mlflow()

def load_ml_model_and_vectorizer(model_name: str, version: int = None):
    """Loads the ML model and TF-IDF vectorizer from MLflow and local path."""
    client = MlflowClient()
    if version is None:
        model_uri = f"models:/{model_name}/latest"
    else:
        model_uri = f"models:/{model_name}/{version}"
    
    logger.info(f"Attempting to load ML model from MLflow: {model_uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"ML model '{model_name}' loaded successfully from MLflow (version: {version if version else 'latest'}).")
    except Exception as e:
        logger.critical(f"FATAL: Failed to load ML model '{model_name}' from MLflow: {e}")
        logger.exception("MLflow model loading traceback:")
        print(f"FATAL ERROR: MLflow model '{model_name}' loading failed: {e}", file=sys.stderr)
        sys.exit(1) # Crash if ML model cannot be loaded

    # Vectorizer path (relative to the app's root directory)
    vectorizer_path = os.path.join("models", "vectorizers", "tfidf3gram_vectorizer.pkl")
    if not os.path.exists(vectorizer_path):
        logger.critical(f"FATAL: TF-IDF vectorizer not found at expected path: {vectorizer_path}")
        print(f"FATAL ERROR: TF-IDF vectorizer not found at: {vectorizer_path}", file=sys.stderr)
        sys.exit(1) # Crash if vectorizer is not found
    
    logger.info(f"Attempting to load TF-IDF vectorizer from: {vectorizer_path}")
    try:
        vectorizer = joblib.load(vectorizer_path)
        logger.info("TF-IDF vectorizer loaded successfully.")
    except Exception as e:
        logger.critical(f"FATAL: Failed to load TF-IDF vectorizer from {vectorizer_path}: {e}")
        logger.exception("Vectorizer loading traceback:")
        print(f"FATAL ERROR: TF-IDF vectorizer loading failed: {e}", file=sys.stderr)
        sys.exit(1) # Crash if vectorizer cannot be loaded
        
    return model, vectorizer

# Load ML model and vectorizer once at application startup
model_sentiment, vectorizer_sentiment = None, None # Initialize to None
try:
    model_sentiment, vectorizer_sentiment = load_ml_model_and_vectorizer("lightGBM_model_v2", version=None)
    logger.info("All machine learning models and vectorizers initialized successfully for sentiment analysis.")
    print("DEBUG: All ML models for sentiment analysis loaded successfully.", file=sys.stderr)
except Exception as e:
    logger.critical(f"FATAL: Unhandled exception during ML model/vectorizer startup: {e}")
    logger.exception("Startup ML model/vectorizer traceback:")
    print(f"FATAL ERROR: Unhandled exception during ML model/vectorizer startup: {e}", file=sys.stderr)
    sys.exit(1)


# -----------------------------------------------------------------------------
# Gemini API Key Configuration (SINGLE DEFINITION - happens once on app startup)
# -----------------------------------------------------------------------------
logger.debug("Attempting to load GEMINI_API_KEY from environment variables...")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    logger.critical("!!!!!!! FATAL: GEMINI_API_KEY environment variable is NOT SET !!!!!!!")
    print("FATAL ERROR: GEMINI_API_KEY NOT SET! Please set it as a GitHub Secret.", file=sys.stderr)
    sys.exit(1) # Crash the app if the key is missing (essential for /gemini endpoint)
else:
    logger.info("GEMINI_API_KEY successfully loaded (first 5 chars): %s", gemini_api_key[:5])
    try:
        genai.configure(api_key=gemini_api_key)
        logger.info("Gemini API client configured successfully at app startup.")
        print("DEBUG: Gemini API client configured successfully.", file=sys.stderr)
    except Exception as e:
        logger.critical(f"FATAL: Failed to configure Gemini API client during startup: {e}")
        logger.exception("Gemini API client configuration traceback:")
        print(f"FATAL ERROR: Gemini client configuration failed: {e}", file=sys.stderr)
        sys.exit(1) # Crash if Gemini client cannot be configured


# -----------------------------------------------------------------------------
# Preprocessing Function (Centralized)
# -----------------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    """Clean and preprocess the input text for sentiment analysis."""
    try:
        text = text.strip().lower()
        text = url_pattern.sub('', text)  # Remove URLs
        text = text.replace('\n', ' ')  # Replace newlines with spaces
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove all non-ASCII characters
        # Allow punctuation for initial regex, then remove it cleanly
        text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text) # Keep basic punctuation for now before full removal
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text) # Remove punctuation after main cleaning

        words = [word for word in text.split() if word not in stop_words] # Remove stop words
        lemmatized = [lemmatizer.lemmatize(word) for word in words] # Lemmatize words
        cleaned_text = ' '.join(lemmatized)
        return cleaned_text
    except Exception as e:
        logger.error(f"Error in preprocess_text function: {e}")
        print(f"DEBUG: Error in preprocess_text: {e}", file=sys.stderr)
        return "" # Return empty string on error


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.route('/', methods=['GET'])
def home():
    """Basic health check endpoint."""
    logger.info("Received GET request to / endpoint (health check).")
    print("DEBUG: GET request received at / endpoint.", file=sys.stderr)
    return "YouTube Comment Analysis Flask app is running!", 200

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict sentiment of a list of comments."""
    logger.info("Received POST request to /predict endpoint.")
    print("DEBUG: POST request received at /predict endpoint.", file=sys.stderr)

    data = request.get_json()
    if not data or 'comments' not in data or not isinstance(data['comments'], list):
        logger.warning("Invalid payload for /predict: 'comments' key missing or not a list.")
        print("DEBUG: Invalid payload for /predict: 'comments' key missing or not a list.", file=sys.stderr)
        return jsonify({'error': 'Invalid payload. Expecting {"comments": ["comment1", "comment2"]}'}), 400

    comments = data.get('comments')
    
    # Ensure ML models are loaded before proceeding
    if model_sentiment is None or vectorizer_sentiment is None:
        logger.critical("ML models are not loaded. Cannot process /predict request.")
        return jsonify({"error": "Server not ready: ML models not loaded."}), 503

    try:
        cleaned_comments = [preprocess_text(comment) for comment in comments]
        # Filter out empty strings from preprocessing that might cause issues with vectorizer
        non_empty_cleaned_comments = [c for c in cleaned_comments if c.strip()]
        
        if not non_empty_cleaned_comments:
            logger.warning("No valid comments for prediction after preprocessing.")
            return jsonify({"message": "No valid comments to predict, all were empty after preprocessing."}), 200 # Or 400, depending on desired behavior for empty input

        transformed_comments = vectorizer_sentiment.transform(non_empty_cleaned_comments)
        
        # Ensure feature names alignment, if vectorizer changes shape, it will be caught here
        # This part is complex due to MLflow loading. Simpler to rely on columns in DataFrame.
        # For joblib loaded vectorizer, get_feature_names_out() is standard.
        try:
            feature_names = vectorizer_sentiment.get_feature_names_out()
            df_transformed_comments = pd.DataFrame(transformed_comments.toarray(), columns=feature_names)
        except AttributeError: # Fallback if get_feature_names_out is not available
            num_features = transformed_comments.shape[1]
            df_transformed_comments = pd.DataFrame(transformed_comments.toarray(), columns=[f'feature_{i}' for i in range(num_features)])
        
        prediction = model_sentiment.predict(df_transformed_comments).tolist()

        # Map predictions back to original comments, handle cases where comments became empty
        response_data = []
        pred_idx = 0
        for i, original_comment in enumerate(comments):
            if cleaned_comments[i].strip(): # Only add if it was successfully processed
                sentiment_val = prediction[pred_idx]
                response_data.append({"comment": original_comment, "sentiment": sentiment_val})
                pred_idx += 1
            else:
                response_data.append({"comment": original_comment, "sentiment": "unpredictable_empty_after_cleaning"}) # Indicate it was skipped

        logger.info("Successfully processed /predict request.")
        print("DEBUG: /predict request processed successfully.", file=sys.stderr)
        return jsonify(response_data)
    except Exception as e:
        logger.critical(f"Unhandled error during /predict endpoint processing: {e}")
        logger.exception("Full traceback for /predict endpoint:")
        print(f"DEBUG: Unhandled error in /predict: {type(e).__name__}: {e}", file=sys.stderr)
        return jsonify({"error": f"Internal server error during prediction: {e}"}), 500


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    logger.info("Received POST request to /generate_chart endpoint.")
    print("DEBUG: POST request received at /generate_chart endpoint.", file=sys.stderr)
    sentiment_counts = request.json.get("sentiment_counts", {})
    labels = ['Neutral', 'Positive', 'Negative']
    # Ensure counts are integers
    sizes = [
        int(sentiment_counts.get('0', 0)),
        int(sentiment_counts.get('1', 0)),
        int(sentiment_counts.get('2', 0))
    ]
    
    if sum(sizes) == 0:
        logger.warning("No sentiment data provided for chart generation.")
        print("DEBUG: No sentiment data for chart generation.", file=sys.stderr)
        return jsonify({"error": "No sentiment data for chart generation. Expected 'sentiment_counts'."}), 400

    try:
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
        plt.close(fig) # Close the figure to free memory
        logger.info("Chart generated successfully.")
        print("DEBUG: Chart generated successfully.", file=sys.stderr)
        return send_file(output, mimetype='image/png')
    except Exception as e:
        logger.critical(f"Error generating chart: {e}")
        logger.exception("Full traceback for /generate_chart:")
        print(f"DEBUG: Error generating chart: {type(e).__name__}: {e}", file=sys.stderr)
        return jsonify({"error": f"Failed to generate chart: {e}"}), 500


@app.route("/generate_trend_graph", methods=["POST"])
def generate_trend_graph():
    logger.info("Received POST request to /generate_trend_graph endpoint.")
    print("DEBUG: POST request received at /generate_trend_graph endpoint.", file=sys.stderr)
    try:
        sentiment_data = request.json.get("sentiment_data", [])
        if not sentiment_data:
            logger.warning("No sentiment data provided for trend graph.")
            print("DEBUG: No sentiment data for trend graph.", file=sys.stderr)
            return jsonify({"error": "No sentiment data provided. Expected 'sentiment_data'."}), 400

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)

        if df.empty:
            logger.warning("No valid timestamps in sentiment data after cleaning for trend graph.")
            print("DEBUG: No valid timestamps for trend graph.", file=sys.stderr)
            return jsonify({"error": "No valid timestamps in provided sentiment data."}), 400

        df['month'] = df['timestamp'].dt.to_period('M').dt.to_timestamp()
        # Ensure sentiment column is correctly mapped
        df['sentiment'] = df['sentiment'].map({0: "Neutral", 1: "Positive", 2: "Negative"})
        df['count'] = 1

        trend = df.groupby(['month', 'sentiment'])['count'].count().unstack().fillna(0)
        
        if trend.empty:
            logger.warning("Trend DataFrame is empty after grouping. No data to plot.")
            print("DEBUG: Trend DataFrame is empty after grouping.", file=sys.stderr)
            return jsonify({"error": "Not enough data to generate trend graph after grouping."}), 400


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
        plt.close(fig) # Close the figure to free memory
        logger.info("Trend graph generated successfully.")
        print("DEBUG: Trend graph generated successfully.", file=sys.stderr)
        return send_file(output, mimetype="image/png")

    except Exception as e:
        logger.critical(f"Error generating trend graph: {e}")
        logger.exception("Full traceback for /generate_trend_graph:")
        print(f"DEBUG: Error generating trend graph: {type(e).__name__}: {e}", file=sys.stderr)
        return jsonify({"error": f"Failed to generate trend graph: {e}"}), 500


@app.route('/gemini', methods=['POST'])
def gemini_summary():
    """Endpoint to summarize YouTube video transcripts using Gemini API."""
    logger.debug("Received POST request to /gemini endpoint.")
    print("DEBUG: Request received at /gemini endpoint.", file=sys.stderr)

    data = request.json
    video_id = data.get('video_id')

    if not video_id:
        logger.warning("Missing 'video_id' in /gemini request payload.")
        print("DEBUG: Missing video_id in /gemini request.", file=sys.stderr)
        return jsonify({"error": "Missing video_id"}), 400
    
    # Ensure Gemini API key is configured before proceeding
    if genai.models is None: # Simple check if genai is configured
        logger.critical("Gemini API client not configured. Cannot process /gemini request.")
        return jsonify({"error": "Server not ready: Gemini API not configured."}), 503

    try:
        logger.info("Attempting to get YouTube transcript for video_id: %s", video_id)
        print(f"DEBUG: Attempting transcript for video_id: {video_id}", file=sys.stderr)

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        transcript = None
        try:
            transcript = transcript_list.find_transcript(['en', 'en-US'])
            logger.info("Found specific English transcript.")
            print("DEBUG: Found specific English transcript.", file=sys.stderr)
        except NoTranscriptFound:
            logger.warning("No specific English transcript found for %s, trying auto-generated.", video_id)
            print(f"DEBUG: No specific English transcript for {video_id}, trying auto-generated.", file=sys.stderr)
            try:
                transcript = transcript_list.find_generated_transcript(['en'])
                logger.info("Found auto-generated English transcript.")
                print("DEBUG: Found auto-generated English transcript.", file=sys.stderr)
            except NoTranscriptFound:
                # If even auto-generated fails, re-raise to hit the outer except
                raise NoTranscriptFound(f"No English (manual or auto-generated) transcript found for video_id: {video_id}")

        if transcript is None: # Safeguard
             raise NoTranscriptFound(f"Transcript object is None after search attempts for video_id: {video_id}")

        full_transcript = " ".join([d['text'] for d in transcript.fetch()])
        logger.info("Successfully fetched YouTube transcript for video_id: %s (Length: %d)", video_id, len(full_transcript))
        print(f"DEBUG: Successfully fetched transcript for {video_id}, length: {len(full_transcript)}", file=sys.stderr)

        logger.info("Attempting to generate Gemini summary for video_id: %s", video_id)
        print(f"DEBUG: Calling Gemini API for video_id: {video_id}", file=sys.stderr)

        model_gemini = genai.GenerativeModel('gemini-pro') # Renamed to avoid potential conflict with ML model
        prompt = f"Summarize the following YouTube video transcript:\n\n{full_transcript}"
        response_gemini = model_gemini.generate_content(prompt) # Renamed to avoid potential conflict
        summary = response_gemini.text
        logger.info("Successfully generated Gemini summary for video_id: %s (Length: %d)", video_id, len(summary))
        print(f"DEBUG: Gemini summary generated for video_id: {video_id}, length: {len(summary)}", file=sys.stderr)

        return jsonify({"video_id": video_id, "summary": summary}), 200

    except NoTranscriptFound as e:
        logger.warning("No transcript found for video_id: %s. Error: %s", video_id, e)
        print(f"DEBUG: !!! CAUGHT NoTranscriptFound for video_id: {video_id} !!! Error: {e}", file=sys.stderr)
        return jsonify({"error": f"No transcript found for this video: {e}"}), 404
    except TranscriptsDisabled as e:
        logger.warning("Transcripts are disabled for video_id: %s. Error: %s", video_id, e)
        print(f"DEBUG: !!! CAUGHT TranscriptsDisabled for video_id: {video_id} !!! Error: {e}", file=sys.stderr)
        return jsonify({"error": f"Transcripts are disabled for this video: {e}"}), 403
    except Exception as e:
        logger.critical(f"!!! UNHANDLED EXCEPTION in /gemini endpoint for video_id: {video_id} !!! Error: {e}")
        logger.exception("Full traceback for /gemini:") # This should print the stack trace
        print(f"DEBUG: !!! CAUGHT UNHANDLED EXCEPTION in /gemini: {type(e).__name__}: {e} !!!", file=sys.stderr)
        return jsonify({"error": f"Internal server error processing Gemini request: {e}"}), 500


@app.route('/generate_wordcloud', methods=['POST'])
def wordcloud():
    logger.info("Received POST request to /generate_wordcloud endpoint.")
    print("DEBUG: POST request received at /generate_wordcloud endpoint.", file=sys.stderr)
    comments = request.json.get("comments", [])
    if not comments:
        logger.warning("No comments provided for wordcloud generation.")
        print("DEBUG: No comments for wordcloud.", file=sys.stderr)
        return jsonify({"error": "No comments provided for wordcloud generation."}), 400

    text = " ".join(preprocess_text(comment) for comment in comments) # Use preprocess_text here
    if not text.strip():
        logger.warning("No valid text after preprocessing for wordcloud generation.")
        print("DEBUG: No valid text after preprocessing for wordcloud.", file=sys.stderr)
        return jsonify({"error": "No valid text after cleaning for wordcloud."}), 400

    try:
        wordcloud_img = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud_img, interpolation='bilinear')
        ax.axis('off')
        output = BytesIO()
        plt.savefig(output, format='png', bbox_inches='tight')
        output.seek(0)
        plt.close(fig) # Close the figure to free memory
        logger.info("Wordcloud generated successfully.")
        print("DEBUG: Wordcloud generated successfully.", file=sys.stderr)
        return send_file(output, mimetype='image/png')
    except Exception as e:
        logger.critical(f"Error generating wordcloud: {e}")
        logger.exception("Full traceback for /generate_wordcloud:")
        print(f"DEBUG: Error generating wordcloud: {type(e).__name__}: {e}", file=sys.stderr)
        return jsonify({"error": f"Failed to generate wordcloud: {e}"}), 500

# -----------------------------------------------------------------------------
# Run App - This block is NOT for Gunicorn. It's for local direct execution.
# Gunicorn imports 'app' directly.
# -----------------------------------------------------------------------------
# if __name__ == '__main__':
#     # app.run(debug=True) # NEVER use debug=True in production. Set to False for local testing.
#     app.run(host='0.0.0.0', port=5000)
