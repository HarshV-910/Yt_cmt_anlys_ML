# tests/test_model.py
import mlflow
import pytest
from src.config.mlflow_config import setup_mlflow
setup_mlflow()

def test_staging_model_loaded():
    client = mlflow.MlflowClient()
    try:
        model_version = client.get_model_version_by_alias("lightGBM_model_v2", "Staging")
        model_uri = f"models:/lightGBM_model_v2@Staging"
        model = mlflow.pyfunc.load_model(model_uri)
        assert model is not None
    except Exception as e:
        pytest.fail(f"Staging model could not be loaded: {e}")
