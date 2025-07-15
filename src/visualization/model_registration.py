import os
import json
import logging
import mlflow
from mlflow.exceptions import MlflowException

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

logger = logging.getLogger('model_registration_logger')
logger.setLevel(logging.DEBUG)
os.makedirs('reports/logs', exist_ok=True)
file_handler = logging.FileHandler('reports/logs/model_registration.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# -----------------------------------------------------------------------------
# MLflow Tracking URI (uncomment for local)
# -----------------------------------------------------------------------------
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# -----------------------------------------------------------------------------
# Load Experiment Info
# -----------------------------------------------------------------------------

def load_experiment_info(info_path: str) -> dict:
    try:
        with open(info_path, 'r') as f:
            info = json.load(f)
        logger.info(f"Experiment info loaded from {info_path}")
        return info
    except Exception as e:
        logger.error(f"Failed to load experiment info: {e}")
        raise

# -----------------------------------------------------------------------------
# Register Model to MLflow Model Registry
# -----------------------------------------------------------------------------

def register_model(info: dict):
    try:
        run_id = info['run_id']
        model_name = info['model_name']
        model_uri = f"runs:/{run_id}/{model_name}"

        logger.info(f"Registering model from {model_uri} as {model_name}")
        # Register the model
        result = mlflow.register_model(model_uri, model_name)
        logger.info(f"Model registered as version {result.version}.")


        # client = mlflow.tracking.MlflowClient()
        # # Transition to Staging
        # client.transition_model_version_stage(
        #     name=model_name,
        #     version=result.version,
        #     stage="Staging",
        #     archive_existing_versions=True
        # )
        # logger.info(f"Model version {result.version} transitioned to Staging.")
        # client.set_registered_model_alias(model_name, "staging", result.version)

        logger.debug(f"Model {model_name} registered with version {result.version}")


    except MlflowException as e:
        logger.error(f"MLflow error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during model registration: {e}")
        raise

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main():
    try:
        info_path = os.path.join("reports", "experiment_info.json")
        info = load_experiment_info(info_path)
        register_model(info)
        logger.info("Model registration pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"Model registration pipeline failed: {e}")
        raise

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
