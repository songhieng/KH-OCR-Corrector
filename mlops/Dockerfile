FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed model vector_db

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_NAME="paraphrase-multilingual-mpnet-base-v2"
ENV MODEL_DIR="/app/model"
ENV VECTOR_DB_PATH="/app/vector_db/khmer_names.pkl"
ENV DEVICE="cpu"

# Expose ports
EXPOSE 8000

# Default command to run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Usage:
# For API server: docker run -p 8000:8000 -e NAMES_FILE=/app/data/processed/khmer_names.txt khmer-corrector
# For Streamlit: docker run -p 8501:8501 -e NAMES_FILE=/app/data/processed/khmer_names.txt khmer-corrector streamlit run app/app.py 