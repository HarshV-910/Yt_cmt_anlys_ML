# tests/test_model_signature.py
import mlflow
import pytest
import os
import joblib
import pandas as pd
from src.config.mlflow_config import setup_mlflow
setup_mlflow()


def test_model_signature():
    vectorizer_path = os.path.join("models", "vectorizers", "tfidf3gram_vectorizer.pkl")
    vectorizer = joblib.load(vectorizer_path)
    model_uri = "models:/lightGBM_model_v2@Staging"
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Model loaded from {model_uri} with signature: {model.metadata.signature}")

    input_example = "Hello, this is a test comment for sentiment analysis."
    input_example_transformed = vectorizer.transform([input_example])
    input_df = pd.DataFrame(input_example_transformed.toarray(), columns=vectorizer.get_feature_names_out())

    prediction = model.predict(input_df)
    assert input_df.shape[1] == len(vectorizer.get_feature_names_out()), "Input feature count mismatch."
    assert len(prediction) == input_df.shape[0], "Output prediction count mismatch."
