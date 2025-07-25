import os
from dotenv import load_dotenv
import mlflow

load_dotenv()  # Loads from .env file

def setup_mlflow():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    # mlflow.set_tracking_uri("http://ec2-51-20-37-158.eu-north-1.compute.amazonaws.com:5000") # uri in .env looks like this
    # GEMINI_API_KEY = "AIzaSyDStfTskjdfhkajfd"  # Replace with your key


def setup_gemini():
    return os.getenv("GEMINI_API_KEY")