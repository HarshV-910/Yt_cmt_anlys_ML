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
# Utility Functions
# -----------------------------------------------------------------------------
def clean_text(text):
    text = text.strip().lower()
    text = url_pattern.sub('', text)
    text = text.replace('\n', ' ')
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    words = [word for word in text.split() if word not in stop_words]
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized)

def dummy_sentiment(comment):
    """Mock sentiment model for demo."""
    cleaned = clean_text(comment)
    if not cleaned:
        return 0  # Neutral
    return random.choices([0, 1, 2], weights=[0.3, 0.5, 0.2])[0]  # 0: neutral, 1: positive, 2: negative

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get("comments", [])
    result = []
    for comment in comments:
        sentiment = dummy_sentiment(comment)
        result.append({"comment": comment, "sentiment": sentiment})
    return jsonify(result)

# @app.route('/generate_chart', methods=['POST'])
# def generate_chart():
#     sentiment_counts = request.json.get("sentiment_counts", {})
#     labels = ['Neutral', 'Positive', 'Negative']
#     sizes = [
#         sentiment_counts.get('0', 0),
#         sentiment_counts.get('1', 0),
#         sentiment_counts.get('2', 0)
#     ]
#     fig, ax = plt.subplots()
#     ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
#     ax.axis('equal')
#     output = BytesIO()
#     plt.savefig(output, format='png')
#     output.seek(0)
#     return send_file(output, mimetype='image/png')

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


# @app.route('/generate_trend_graph', methods=['POST'])
# def generate_trend():
#     data = request.json.get("sentiment_data", [])
#     df = pd.DataFrame(data)
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     df.set_index('timestamp', inplace=True)
#     df['sentiment'] = df['sentiment'].map({0: "Neutral", 1: "Positive", 2: "Negative"})
#     df['count'] = 1
#     trend = df.groupby([pd.Grouper(freq='H'), 'sentiment']).count().unstack().fillna(0)
#     # df['month'] = df['timestamp'].dt.to_period('M').dt.to_timestamp()
#     # df['sentiment'] = df['sentiment'].map({0: "Neutral", 1: "Positive", 2: "Negative"})
#     # df['count'] = 1
#     # trend = df.groupby(['month', 'sentiment'])['count'].count().unstack().fillna(0)

#     trend.columns = trend.columns.droplevel(0)
#     fig, ax = plt.subplots(figsize=(6, 3))
#     trend.plot(ax=ax)
#     plt.tight_layout()
#     output = BytesIO()
#     plt.savefig(output, format='png')
#     output.seek(0)
#     return send_file(output, mimetype='image/png')

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
