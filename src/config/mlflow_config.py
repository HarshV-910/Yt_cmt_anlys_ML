import os
from dotenv import load_dotenv
import mlflow

load_dotenv()  # Loads from .env file

def setup_mlflow():
    # mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_tracking_uri("http://ec2-51-20-85-228.eu-north-1.compute.amazonaws.com:5000") # uri in .env looks like this


def setup_gemini():
    # return os.getenv("GEMINI_API_KEY")
    return "AIzaSyDStfTRZ2MuOXzH-00_21KegNppcMVmcJc"