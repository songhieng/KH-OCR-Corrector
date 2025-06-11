# Khmer Name OCR Corrector

A system for correcting OCR errors in Khmer names using both semantic matching and Latin/English transliteration matching.

## Overview

Tesseract OCR often struggles with Khmer script, resulting in inaccurate name recognition. This system provides a two-stage matching approach:

1. **First stage**: Semantic and character-based matching for Khmer text
2. **Second stage**: Fuzzy matching of Latin/English transliterations 

By combining both approaches, we can achieve more accurate name correction even when the semantic search returns multiple similar matches.

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
│   └── README.md            # Deployment guide
├── README.md                # Main project documentation
└── requirements.txt         # Project dependencies
```

## Features

- Semantic search using sentence transformers (multilingual-mpnet-base-v2)
- Character-based matching as a fallback
- Latin/English transliteration matching using FuzzyWuzzy
- FastAPI backend for integration
- Streamlit UI for easy interaction
- Docker support for deployment
- Persistent vector database for improved performance

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/khmer-name-ocr-corrector.git
cd khmer-name-ocr-corrector

# Install dependencies
pip install -r requirements.txt
```

### Preparing Data

Create a text file with Khmer names and their Latin transliterations in the format:

```
songhieng,សុងហៀង
dara,ដារា
sok,សុខ
vichet,វិចិត្រ
```

Save this file to `data/raw/khmer_names.txt`.

Process the data:

```bash
python -c "from utils.preprocess import prepare_dataset; prepare_dataset('data/raw/khmer_names.txt', 'data/processed')"
```

### Running the Streamlit App

```bash
streamlit run app/app.py
```

Visit http://localhost:8501 in your browser.

### Running the API

```bash
export NAMES_FILE="data/processed/khmer_names.txt"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Visit http://localhost:8000/docs to see the API documentation.

## Using the API

### Python Client

```python
from integration.python_client import KhmerNameCorrectorClient

client = KhmerNameCorrectorClient("http://localhost:8000")

# Match a single name
result = client.match("សុងហៀង", "songhieng")
print(result)

# Batch matching
batch_items = [
    {"khmer_text": "សុងហៀង", "latin_text": "songhieng"},
    {"khmer_text": "ដារា", "latin_text": "dara"}
]
batch_results = client.batch_match(batch_items)
print(batch_results)
```

## Docker Deployment

```bash
cd mlops
docker-compose up -d
```

For more detailed deployment instructions, see the [MLOps README](mlops/README.md).

## Configuration

Key environment variables:

- `MODEL_NAME`: Name of the sentence transformer model (default: `paraphrase-multilingual-mpnet-base-v2`)
- `NAMES_FILE`: Path to the processed names file
- `VECTOR_DB_PATH`: Path to save/load vector database
- `DEVICE`: Device to use for model inference (`cpu` or `cuda`)

## License

MIT

## Acknowledgements

- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [FuzzyWuzzy](https://github.com/seatgeek/fuzzywuzzy)

## Testing

The project includes a comprehensive test suite located in the `tests` directory. For detailed information on running tests, see [tests/README_testing.md](tests/README_testing.md).

To run all tests:

```bash
python run_tests.py
```

To run specific test categories:

```bash
python run_tests.py --test test_utils
``` 