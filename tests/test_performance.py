# tests/test_performance.py --> currently i am comment out it because it is taking load and cost on AWS 
import mlflow
import pytest
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score
from src.visualization.model_evaluation import load_test_data  # adjust import as per your structure
from src.config.mlflow_config import setup_mlflow
setup_mlflow()


THRESHOLD = 0.75
MODEL_NAME = "lightGBM_model_v2"


def test_model_performance():
    vectorizer_path = os.path.join("models", "vectorizers", "tfidf3gram_vectorizer.pkl")
    vectorizer = joblib.load(vectorizer_path)
    
    client = mlflow.MlflowClient()
    interim_path = os.path.join("data", "interim")
    test_df = load_test_data(interim_path)
    X_test_raw = test_df['clean_comment']
    y_test = test_df['category']
    y_test = y_test.map({-1: 2, 0: 0, 1: 1})

    # Transform the text data using the vectorizer
    X_test = vectorizer.transform(X_test_raw)
    
    # Convert sparse matrix to dense if needed (some models expect dense arrays)
    if hasattr(X_test, 'toarray'):
        X_test = X_test.toarray()

    X_test = pd.DataFrame(X_test, columns=vectorizer.get_feature_names_out())

    # Load staging model
    staging_version = client.get_model_version_by_alias(MODEL_NAME, "Staging")
    staging_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@Staging")
    staging_preds = staging_model.predict(X_test)
    staging_acc = accuracy_score(y_test, staging_preds)

    assert staging_acc >= THRESHOLD, f"Staging model accuracy {staging_acc} below threshold {THRESHOLD}"

    # Check if there's a production model
    prod_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    prod_version = None
    max_prod_acc = 0.0

    for v in prod_versions:
        if "Production" in v.aliases:
            prod_version = v
            break

    if prod_version:
        prod_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@Production")
        prod_preds = prod_model.predict(X_test)
        prod_acc = accuracy_score(y_test, prod_preds)

        if staging_acc > prod_acc:
            client.set_registered_model_alias(MODEL_NAME, "Production", staging_version.version)
            client.delete_registered_model_alias(MODEL_NAME, "Production")
            client.set_registered_model_alias(MODEL_NAME, "Production", staging_version.version)

            for v in prod_versions:
                if "Production" in v.aliases and v.version != staging_version.version:
                    client.set_registered_model_alias(MODEL_NAME, "Archived", v.version)
        else:
            print("Staging model not promoted. Previous Production is better.")
    else:
        # First model ever
        client.set_registered_model_alias(MODEL_NAME, "Production", staging_version.version)
