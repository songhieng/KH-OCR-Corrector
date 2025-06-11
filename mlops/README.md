# Khmer Name OCR Corrector - MLOps Guide

This guide explains how to deploy, monitor, and maintain the Khmer Name OCR Corrector system in a production environment.

## Project Structure

```
/
├── app/                     # Streamlit web application
│   └── app.py               # Streamlit UI code
├── api/                     # API service
│   ├── main.py              # FastAPI application
│   └── routes/              # API route definitions
├── data/                    # Data storage
│   ├── raw/                 # Original name files
│   └── processed/           # Processed name files
├── model/                   # Model storage
│   ├── embedding.py         # Model utilities
│   └── <downloaded_models>/ # Downloaded models from HuggingFace
├── utils/                   # Utility functions
│   ├── matcher.py           # Name matching logic
│   └── preprocess.py        # Data preprocessing
├── integration/             # Integration examples
│   ├── python_client.py     # Python client for API
│   └── examples/            # Example scripts
├── vector_db/               # Local vector storage
│   └── <vector_files>.pkl   # Saved vector databases
├── mlops/                   # MLOps configuration
│   ├── Dockerfile           # Container definition
│   ├── docker-compose.yml   # Multi-container setup
│   └── README.md            # This guide
├── README.md                # Main project documentation
└── requirements.txt         # Project dependencies
```

## Environment Setup

### Local Development

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Set up environment variables:

```bash
# For the API
export MODEL_NAME="paraphrase-multilingual-mpnet-base-v2"
export MODEL_DIR="./model"
export VECTOR_DB_PATH="./vector_db/khmer_names.pkl"
export NAMES_FILE="./data/processed/khmer_names.txt"
export DEVICE="cpu"  # or "cuda" for GPU

# For the Streamlit app
export MODEL_DIR="./model"
export VECTOR_DB_DIR="./vector_db"
```

### Docker Deployment

1. Build and start containers:

```bash
cd mlops
docker-compose up -d
```

2. Services available:
   - API: http://localhost:8000
   - Streamlit app: http://localhost:8501
   - API documentation: http://localhost:8000/docs

## Workflow

### Data Preparation

1. Place your raw names file in `data/raw/`
2. Process the data:

```bash
python -c "from utils.preprocess import prepare_dataset; prepare_dataset('data/raw/names.txt', 'data/processed', split=True)"
```

### Model Management

1. Download a model:

```bash
python -c "from model.embedding import download_model; download_model('paraphrase-multilingual-mpnet-base-v2', 'model')"
```

2. Run the system with a specific model:

```bash
export MODEL_NAME="paraphrase-multilingual-mpnet-base-v2"
```

### Vector Database Management

The system will automatically create and save vector databases when first processing a names file. To force rebuilding:

```bash
rm -f vector_db/*.pkl
```

## Running Services

### API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Streamlit App

```bash
streamlit run app/app.py
```

## Monitoring and Maintenance

### Logs

- API logs: Check the console output or redirect to a file
- Streamlit logs: Check the console output

### Performance Monitoring

Key metrics to monitor:
- API response time
- Memory usage (especially with large models)
- GPU memory (if using GPU)
- Number of API requests

### Regular Maintenance

1. Update models when new versions are available:

```bash
python -c "from model.embedding import download_model; download_model('model_name', 'model', force_download=True)"
```

2. Rebuild vector databases when new data is available:

```bash
rm -f vector_db/*.pkl
```

## Scaling

### Horizontal Scaling

The API service can be scaled horizontally behind a load balancer. The Streamlit app can also be scaled, but each instance needs access to the shared data and model files.

### Performance Optimization

- Use a GPU for faster inference
- Use the smaller `paraphrase-multilingual-MiniLM-L12-v2` model for better performance with a small quality trade-off
- Pre-compute vector databases for all your datasets

## Troubleshooting

### Common Issues

1. **Out of memory errors**: Reduce batch size or use a smaller model
2. **Slow inference**: Ensure you're using GPU if available, or consider a smaller model
3. **API errors**: Check the logs for details

### Debugging

Enable debug logs by setting:

```bash
export LOG_LEVEL=DEBUG
```

## Security Considerations

1. Modify CORS settings in `api/main.py` for production
2. Add authentication to the API for production use
3. Use HTTPS for all services in production
4. Consider using API keys for client authentication

## Backup and Recovery

1. Regularly backup:
   - Data files in `data/`
   - Vector databases in `vector_db/`

2. Recovery:
   - Restore data files
   - Models can be re-downloaded from HuggingFace
   - Vector databases will be rebuilt automatically if missing 