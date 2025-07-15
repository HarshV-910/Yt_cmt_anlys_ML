import os
import logging
import pandas as pd
import pickle
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
# from sklearn.metrics import accuracy_score

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

logger = logging.getLogger('model_building_logger')
logger.setLevel(logging.DEBUG)
os.makedirs('reports/logs', exist_ok=True)
file_handler = logging.FileHandler('reports/logs/model_building.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# -----------------------------------------------------------------------------
# Load Parameters from YAML
# -----------------------------------------------------------------------------

def load_params(param_path: str) -> dict:
    try:
        with open(param_path, 'r') as file:
            params = yaml.safe_load(file)
            logger.info("Model parameters loaded successfully.")
            return params['model_building']
    except Exception as e:
        logger.error(f"Error loading params from {param_path}: {e}")
        raise

# -----------------------------------------------------------------------------
# Load Processed Data
# -----------------------------------------------------------------------------

def load_data(interim_path: str):
    try:
        train_df = pd.read_csv(os.path.join(interim_path, "train_processed.csv"))
        test_df = pd.read_csv(os.path.join(interim_path, "test_processed.csv"))
        logger.info(f"Loaded train {train_df.shape}, test {test_df.shape} datasets.")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# -----------------------------------------------------------------------------
# Model Training Function
# -----------------------------------------------------------------------------

def train_model(params: dict) -> None:
    try:
        interim_path = os.path.join("data", "interim")
        train_df, test_df = load_data(interim_path)

        X_train_raw = train_df['clean_comment']
        y_train = train_df['category']

        X_test_raw = test_df['clean_comment']
        y_test = test_df['category']

        # Label Encoding if necessary
        y_train = y_train.map({-1:2, 0:0, 1:1})
        y_test = y_test.map({-1:2, 0:0, 1:1})
        logger.info("Labeling applied on target variable.")

        # TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(ngram_range=tuple(params['ngram_range']),
                                     max_features=params['max_features'])
        X_train_vec = vectorizer.fit_transform(X_train_raw)
        X_test_vec = vectorizer.transform(X_test_raw)
        logger.info("TFIDF vectorization completed.")

        # SMOTE Oversampling
        sampler = SMOTE(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X_train_vec, y_train)
        logger.info(f"SMOTE applied. Resampled dataset shape: {X_resampled.shape}")

        # LightGBM Classifier
        model = LGBMClassifier(
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            n_estimators=params['n_estimators'],
            random_state=42
        )

        model.fit(X_resampled, y_resampled)
        logger.info("LightGBM model training completed.")

        return model, vectorizer, sampler
    except Exception as e:
        logger.critical(f"Model building pipeline failed: {e}")
        raise

# -----------------------------------------------------------------------------
# Model Saving
# -----------------------------------------------------------------------------

def save_model(model, output_path):
    """Save trained model as pickle file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as file:
            pickle.dump(model, file)
        logger.info(f"Trained model saved to {output_path}")

        os.makedirs("models/vectorizers", exist_ok=True)
        with open("models/vectorizers/tfidf3gram_vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)

        logger.info("Model, vectorizer saved.")

    except Exception as e:
        logger.error(f"Error saving model to {output_path}: {e}")
        raise

# -----------------------------------------------------------------------------
if __name__ == "__main__":

    param_path = "params.yaml"
    params = load_params(param_path)
    model, vectorizer, sampler= train_model(params)
    save_model(model, "models/model/lightGBM_model_v2.pkl")
    logger.info("Model building pipeline completed successfully.")

