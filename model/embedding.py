"""
Utilities for managing embedding models.
"""
import os
import logging
from typing import Optional, List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import torch
import huggingface_hub

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("embedding")

def download_model(
    model_name: str,
    model_dir: str,
    force_download: bool = False
) -> str:
    """
    Download a model from Hugging Face Hub.
    
    Args:
        model_name: Name of the model to download
        model_dir: Directory to save the model
        force_download: Whether to force re-download if model exists
        
    Returns:
        Path to the downloaded model
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    
    if os.path.exists(model_path) and not force_download:
        logger.info(f"Model already exists at {model_path}")
        return model_path
    
    logger.info(f"Downloading model {model_name} to {model_path}")
    try:
        # Use huggingface_hub to download the model
        huggingface_hub.snapshot_download(
            repo_id=model_name,
            local_dir=model_path,
            local_dir_use_symlinks=False
        )
        logger.info(f"Successfully downloaded model to {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        raise

def load_model(
    model_name_or_path: str,
    model_dir: Optional[str] = None,
    device: str = "cpu",
    download_if_missing: bool = True
) -> SentenceTransformer:
    """
    Load a SentenceTransformer model.
    
    Args:
        model_name_or_path: Name or path of the model
        model_dir: Directory to save/load models
        device: Device to use for model inference
        download_if_missing: Whether to download the model if it's not found locally
        
    Returns:
        Loaded SentenceTransformer model
    """
    # Check if model_name_or_path is a local path
    if os.path.exists(model_name_or_path):
        logger.info(f"Loading model from local path: {model_name_or_path}")
        return SentenceTransformer(model_name_or_path, device=device)
    
    # If model_dir is provided, check if model exists there
    if model_dir:
        model_path = os.path.join(model_dir, model_name_or_path)
        if os.path.exists(model_path):
            logger.info(f"Loading model from: {model_path}")
            return SentenceTransformer(model_path, device=device)
        
        # Download model if it doesn't exist and download_if_missing is True
        if download_if_missing:
            downloaded_path = download_model(model_name_or_path, model_dir)
            logger.info(f"Loading downloaded model from: {downloaded_path}")
            return SentenceTransformer(downloaded_path, device=device)
    
    # If we get here, just try to load directly from Hugging Face
    logger.info(f"Loading model directly from Hugging Face: {model_name_or_path}")
    return SentenceTransformer(model_name_or_path, device=device)

def get_available_models() -> List[Dict[str, str]]:
    """
    Get a list of recommended models for Khmer name matching.
    
    Returns:
        List of dictionaries with model information
    """
    return [
        {
            "name": "paraphrase-multilingual-mpnet-base-v2",
            "description": "State-of-the-art multilingual model, best quality but slower",
            "size": "1.1 GB",
            "languages": "50+ languages including Khmer",
            "url": "https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        },
        {
            "name": "paraphrase-multilingual-MiniLM-L12-v2",
            "description": "Good multilingual model, balanced between quality and speed",
            "size": "420 MB",
            "languages": "50+ languages including Khmer",
            "url": "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        },
        {
            "name": "LaBSE",
            "description": "Language-agnostic BERT Sentence Embedding, good for 109 languages",
            "size": "1.7 GB",
            "languages": "109 languages including Khmer",
            "url": "https://huggingface.co/sentence-transformers/LaBSE"
        }
    ]

def get_device() -> str:
    """
    Get the best available device for model inference.
    
    Returns:
        'cuda' if CUDA is available, 'mps' if MPS is available (Mac with Apple Silicon), otherwise 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'has_mps') and torch.has_mps:
        return "mps"
    else:
        return "cpu" 