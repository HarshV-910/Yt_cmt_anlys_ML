import os
import pandas as pd
import numpy as np
import yaml
import logging
from sklearn.model_selection import train_test_split
import kagglehub

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

logger = logging.getLogger('data_ingestion_logger')
logger.setLevel(logging.DEBUG)

os.makedirs('reports/logs', exist_ok=True)
file_handler = logging.FileHandler('reports/logs/data_ingestion.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def load_params(param_path: str) -> float:
    """Load test_size parameter from YAML."""
    try:
        with open(param_path, 'r') as file:
            params = yaml.safe_load(file)
            test_size = params['data_ingestion']['test_size']
            logger.info(f"Loaded test_size={test_size}")
            return test_size
    except Exception as e:
        logger.error(f"Failed to load params: {e}")
        raise


def download_data(url_path: str,dataset_name: str) -> pd.DataFrame:
    """Download Reddit dataset using kagglehub."""
    try:
        path = kagglehub.dataset_download(url_path)
        reddit_path = os.path.join(path, dataset_name)
        df = pd.read_csv(reddit_path)
        logger.info(f"Data downloaded successfully: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove missing, duplicate, and empty comments."""
    try:
        initial_shape = df.shape
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df = df[df['clean_comment'].astype(str).str.strip() != '']
        logger.info(f"Cleaned data: {initial_shape} → {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        raise


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, save_dir: str) -> None:
    """Save train and test datasets to CSV."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)
        logger.info(f"Saved train ({train_df.shape}) and test ({test_df.shape}) to {save_dir}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise


def main() -> None:
    """Main pipeline execution."""
    try:
        logger.info("Starting data ingestion pipeline.")
        test_size = load_params('params.yaml')
        df = download_data("cosmos98/twitter-and-reddit-sentimental-analysis-dataset", "Reddit_Data.csv")
        df = clean_data(df)
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['category'])
        save_data(train_df, test_df, os.path.join("data", "raw"))
        logger.info("Data ingestion pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
        raise


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    main()

