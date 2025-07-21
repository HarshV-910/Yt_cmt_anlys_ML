# =========================
# Stage 1 — Build stage
# =========================
FROM python:3.10 AS build

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies needed for some Python packages (e.g., LightGBM)
# libgomp1 is crucial for LightGBM's parallel processing.
RUN apt-get update && apt-get install -y libgomp1 --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker's caching
COPY flask_app/requirements.txt .

# Install Python dependencies. --no-cache-dir reduces image size in this stage.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# Create a directory for NLTK data and set environment variable
ENV NLTK_DATA=/usr/local/share/nltk_data 
RUN mkdir -p ${NLTK_DATA}

# Download NLTK data directly using its Python API
RUN python3 -c "import nltk; nltk.download('stopwords', download_dir='${NLTK_DATA}'); nltk.download('wordnet', download_dir='${NLTK_DATA}')"

# --- TEMPORARY DEBUGGING STEPS IN BUILD STAGE ---
# Verify gunicorn executable exists
RUN echo "Checking for gunicorn in build stage:" && \
    ls -la /usr/local/bin/gunicorn || echo "gunicorn executable NOT found in /usr/local/bin"

# Verify NLTK data directory exists and has content
RUN echo "Checking for NLTK data in build stage:" && \
    ls -la /usr/local/share/nltk_data || echo "NLTK_DATA directory NOT found" && \
    ls -la /usr/local/share/nltk_data/corpora || echo "NLTK_DATA corpora NOT found"
# --- END TEMPORARY DEBUGGING STEPS ---

# Copy the Flask application code and all necessary runtime files
COPY flask_app/ /app/flask_app/
COPY src/ /app/src/
COPY models/vectorizers/tfidf3gram_vectorizer.pkl /app/models/vectorizers/tfidf3gram_vectorizer.pkl
COPY gunicorn_config.py /app/

# =========================
# Stage 2 — Final, slim image
# =========================
FROM python:3.10-slim AS final

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install libgomp1 in the FINAL stage (Crucial for LightGBM)
RUN apt-get update && apt-get install -y libgomp1 --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Copy *only* the necessary files from the build stage to the final slim image
COPY --from=build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=build /usr/local/bin/gunicorn /usr/local/bin/
COPY --from=build /usr/local/share/nltk_data /usr/local/share/nltk_data
COPY --from=build /app /app

# Set PYTHONPATH to ensure Python finds your modules
ENV PYTHONPATH=/app:/app/src

# Set Flask environment variables
ENV FLASK_APP=flask_app.app
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Expose application port
EXPOSE 5000

# Set default command to run gunicorn with your application
CMD ["gunicorn", "flask_app.app:app", "-c", "gunicorn_config.py"]



# -------------------------------

# # Use official Python base image
# FROM python:3.10-slim

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# # Set working directory inside the container
# WORKDIR /app

# RUN apt-get update && apt-get install -y libgomp1

# # Copy requirements and install dependencies
# COPY flask_app/requirements.txt .

# RUN pip install --upgrade pip
# RUN pip install --default-timeout=300 -r requirements.txt

# RUN python3 -m nltk.downloader stopwords wordnet
# # Copy your Flask app code
# COPY flask_app/ /app/
# COPY src/ ./src/
# COPY models/vectorizers/tfidf3gram_vectorizer.pkl ./models/vectorizers/tfidf3gram_vectorizer.pkl

# # Set Flask environment variables
# ENV FLASK_APP=app.py
# ENV FLASK_RUN_HOST=0.0.0.0
# ENV FLASK_RUN_PORT=5000

# # Expose port
# EXPOSE 5000

# # Run the app
# CMD ["flask", "run"]
