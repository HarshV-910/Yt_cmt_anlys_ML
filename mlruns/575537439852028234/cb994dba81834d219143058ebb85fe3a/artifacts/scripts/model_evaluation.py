import os
import logging
import json
import joblib
import pickle
import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score


# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

logger = logging.getLogger('model_evaluation_logger')
logger.setLevel(logging.DEBUG)
os.makedirs('reports/logs', exist_ok=True)
file_handler = logging.FileHandler('reports/logs/model_evaluation.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# -----------------------------------------------------------------------------
# MLflow Tracking URI (uncomment for local)
# -----------------------------------------------------------------------------
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Model_Evaluation_lightgbm")

# -----------------------------------------------------------------------------
# Load Test Data
# -----------------------------------------------------------------------------

def load_test_data(interim_path: str) -> pd.DataFrame:
    try:
        test_df = pd.read_csv(os.path.join(interim_path, "test_processed.csv"))
        logger.info(f"Loaded test dataset with shape: {test_df.shape}")
        return test_df
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise

# -----------------------------------------------------------------------------
# Load Model and Vectorizer
# -----------------------------------------------------------------------------

def load_artifacts(model_path: str, vectorizer_path: str):
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        logger.info("Model and vectorizer loaded successfully.")
        return model, vectorizer
    except Exception as e:
        logger.error(f"Error loading model or vectorizer: {e}")
        raise

# -----------------------------------------------------------------------------
# Save Experiment Info
# -----------------------------------------------------------------------------

def save_experiment_info(info: dict, save_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(info, f, indent=4)
        logger.info(f"Experiment info saved to {save_path}")
    except Exception as e:
        logger.error(f"Error saving experiment info: {e}")
        raise

# -----------------------------------------------------------------------------
# Model Evaluation Function
# -----------------------------------------------------------------------------

def evaluate_model():
    try:
        interim_path = os.path.join("data", "interim")
        test_df = load_test_data(interim_path)

        X_test_raw = test_df['clean_comment']
        y_test = test_df['category']
        y_test = y_test.map({-1:2, 0:0, 1:1})

        model_path = os.path.join("models", "model", "lightGBM_model_v2.pkl")
        # os.makedirs(os.path.dirname(model_path), exist_ok=True)
        vectorizer_path = os.path.join("models", "vectorizers", "tfidf3gram_vectorizer.pkl")
        # os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)

        model, vectorizer = load_artifacts(model_path, vectorizer_path)

        X_test_vec = vectorizer.transform(X_test_raw)
        logger.info("TF-IDF transformation completed on test data.")

        with mlflow.start_run(run_name="LightGBM_Evaluation") as run:

            metrics_output_path = os.path.join("reports", "metrics", "metrics.json")
            os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)


            y_pred = model.predict(X_test_vec)
            y_pred_proba = model.predict_proba(X_test_vec)
            print(f"y_test shape: {y_test.shape}")
            print(f"y_pred_proba shape: {y_pred_proba.shape}")


            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')

            logger.info(f"Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
            metrics =  {
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            with open(metrics_output_path, 'w') as file:
                json.dump(metrics, file, indent=4)
            logger.info(f"Metrics saved to {metrics_output_path}")

            for metric, value in metrics.items():
                mlflow.log_metric(metric, float(value))
                logger.info(f"Logged metric: {metric} = {value}")
            
            if hasattr(model, 'get_params'):
                params = model.get_params()
                mlflow.log_params(params)
                logger.info("Logged model parameters.")

            logger.info("Model evaluation pipeline completed successfully.")

            report = classification_report(y_test, y_pred, output_dict=True)
            with open("reports/metrics/classification_report.json", "w") as f:
                json.dump(report, f, indent=4)
            mlflow.log_dict(report, "classification_report.json")
            logger.info("Classification report logged.")


            # # Log metrics to MLflow
            # mlflow.log_metric("accuracy", acc)
            # mlflow.log_metric("f1_score", f1)
            # mlflow.log_metric("precision", precision)
            # mlflow.log_metric("recall", recall)


            # # Ensure directory exists
            # os.makedirs("reports/metrics", exist_ok=True)

            # # Save metrics JSON
            # with open("reports/metrics/metrics.json", "w") as f:
            #     json.dump(metrics, f, indent=4)

            # Log model to MLflow
            mlflow.sklearn.log_model(model, artifact_path="lightGBM_model_v2")
            mlflow.log_artifact(__file__, artifact_path="scripts")
            mlflow.log_artifact(metrics_output_path, artifact_path="metrics")



            # Save experiment info for next stage
            run_id = run.info.run_id
            model_name = "lightGBM_model_v2"
            experiment_info = {
                "model_name": model_name,
                "run_id": run_id
            }

            save_path = os.path.join("reports", "experiment_info.json")
            # with open(save_path, 'w') as file:
            #     json.dump(experiment_info, file, indent=4)
            # logger.info(f"Model info saved to {save_path}")

            save_experiment_info(experiment_info, save_path)
            logger.info("Model evaluation completed and logged to MLflow.")


    except Exception as e:
        logger.critical(f"Model evaluation pipeline failed: {e}")
        raise

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    evaluate_model()



# app.runllm.com/chat/ce1b2868-fbf4-4476-b204-813f48d8dd0c