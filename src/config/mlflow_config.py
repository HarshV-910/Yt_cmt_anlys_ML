import os
from dotenv import load_dotenv
import mlflow

load_dotenv()  # Loads from .env file

def setup_mlflow():
    # mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_tracking_uri("http://ec2-16-171-36-111.eu-north-1.compute.amazonaws.com:5000")
