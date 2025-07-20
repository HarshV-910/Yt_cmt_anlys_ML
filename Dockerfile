# # =========================
# # Stage 1 — Build stage
# # =========================
# FROM python:3.10 AS build

# # Set working directory
# WORKDIR /app

# # to support parallel computing for LightGBM model
# RUN apt-get update && apt-get install -y libgomp1

# # Copy requirements file
# COPY flask_app/requirements.txt .

# # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Download NLTK data
# # RUN python3 -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
# RUN python3 -m nltk.downloader stopwords wordnet

# # Copy source code and model files
# COPY flask_app/ /app/
# COPY src/ ./src/
# COPY models/vectorizers/tfidf3gram_vectorizer.pkl ./models/vectorizers/tfidf3gram_vectorizer.pkl

# # Copy gunicorn config (if needed for runtime)
# COPY gunicorn_config.py .

# # =========================
# # Stage 2 — Final, slim image
# # =========================
# FROM python:3.10-slim AS final

# # Set working directory
# WORKDIR /app

# # Copy everything from build stage
# COPY --from=build /app /app

# # Set Python path to include src directory
# ENV PYTHONPATH=/app
# # ENV PYTHONPATH=/app:/app/src

# # Expose application port
# EXPOSE 5000

# # Set default command to run gunicorn with config
# CMD ["gunicorn", "flask_app.app:app", "-c", "gunicorn_config.py"]


# -------------------------------------------
    
    # # Use Python 3.10 slim for smaller image size
    # FROM python:3.10
    
    # # Set working directory
    # WORKDIR /app
    
    # # Copy requirements file from flask_app directory
    # COPY flask_app/requirements.txt .
    
    # # Install Python dependencies (gunicorn already in requirements.txt)
    # RUN pip install --no-cache-dir -r requirements.txt
    
    # # Download NLTK data
    # RUN python3 -m nltk.downloader stopwords wordnet
    
    # # Copy the entire src directory to maintain import structure
    # COPY flask_app/ /app/
    # # COPY src/ ./src/
    
    # # Copy models directory
    # COPY models/vectorizers/tfidf3gram_vectorizer.pkl ./models/vectorizers/tfidf3gram_vectorizer.pkl

# # Set Python path to include app directory
# ENV PYTHONPATH=/app

# # Expose port 5000
# EXPOSE 5000

# COPY gunicorn_config.py .
# CMD ["gunicorn", "flask_app.app:app", "-c", "gunicorn_config.py"]

# ---------------------------
    
# Use official Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1

# Copy requirements and install dependencies
COPY flask_app/requirements.txt .

RUN pip install --upgrade pip
RUN pip install --default-timeout=300 -r requirements.txt

RUN python3 -m nltk.downloader stopwords wordnet
# Copy your Flask app code
COPY flask_app/ /app/
COPY src/ ./src/
COPY models/vectorizers/tfidf3gram_vectorizer.pkl ./models/vectorizers/tfidf3gram_vectorizer.pkl

# Set Flask environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Expose port
EXPOSE 5000

# Run the app
CMD ["flask", "run"]







# ------------------------------

# # =========================
# # Stage 1 — Build stage
# # =========================
# FROM python:3.10 AS build

# # Set working directory
# WORKDIR /app

# # System-level dependencies (like libgomp for LightGBM, if used)
# RUN apt-get update && apt-get install -y libgomp1

# # Copy and install Python dependencies
# COPY flask_app/requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Download necessary NLTK data
# RUN python3 -m nltk.downloader stopwords wordnet

# # Copy application code and models
# COPY flask_app/ /app/flask_app/
# COPY src/ /app/src/
# COPY models/ /app/models/

# # =========================
# # Stage 2 — Final, runtime stage
# # =========================
# FROM python:3.10-slim AS final

# WORKDIR /app

# # Minimal OS packages (just enough to run Python)
# RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# # Copy from build stage
# COPY --from=build /app /app

# # Set Python path so Flask app and src are importable
# ENV PYTHONPATH=/app

# # Expose the port Flask/Gunicorn will run on
# EXPOSE 5000

# # Default command to run app with Gunicorn
# CMD ["gunicorn", "flask_app.app:app", "-c", "gunicorn_config.py"]
