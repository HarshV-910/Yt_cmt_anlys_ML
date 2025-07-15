import os
import re
import logging
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

logger = logging.getLogger('data_preprocessing_logger')
logger.setLevel(logging.DEBUG)

os.makedirs('reports/logs', exist_ok=True)
file_handler = logging.FileHandler('reports/logs/data_preprocessing.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# -----------------------------------------------------------------------------
# NLTK Data Download
# -----------------------------------------------------------------------------

try:
    nltk.download("stopwords")
    nltk.download("wordnet")
    logger.info("Downloaded NLTK resources.")
except Exception as e:
    logger.error(f"Failed to download NLTK resources: {e}")
    raise

# -----------------------------------------------------------------------------
# Text Cleaning Config
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Text Preprocessing Function
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Data Processing Function
# -----------------------------------------------------------------------------

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    try:
        initial_shape = df.shape
        df.dropna(subset=['clean_comment'], inplace=True)
        df.drop_duplicates(inplace=True)
        df['clean_comment'] = df['clean_comment'].astype(str).apply(clean_text)
        df = df[df['clean_comment'].str.strip() != ''] # remove empty comments
        logger.info(f"Data cleaned: {initial_shape} → {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error preprocessing dataframe: {e}")
        raise

# -----------------------------------------------------------------------------
# Main Execution Function
# -----------------------------------------------------------------------------

def main() -> None:
    try:
        raw_path = os.path.join("data", "raw")
        interim_path = os.path.join("data", "interim")
        os.makedirs(interim_path, exist_ok=True)

        # Load datasets
        train_path = os.path.join(raw_path, "train.csv")
        test_path = os.path.join(raw_path, "test.csv")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logger.info(f"Loaded train {train_df.shape}, test {test_df.shape} datasets.")

        # Preprocess text
        train_df = preprocess_dataframe(train_df)
        test_df = preprocess_dataframe(test_df)

        # Save processed data to data/interim/
        train_df.to_csv(os.path.join(interim_path, "train_processed.csv"), index=False)
        test_df.to_csv(os.path.join(interim_path, "test_processed.csv"), index=False)
        logger.info(f"Processed datasets saved to {interim_path}")

    except Exception as e:
        logger.critical(f"Text preprocessing pipeline failed: {e}")
        raise

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
