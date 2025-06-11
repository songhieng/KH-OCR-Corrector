"""
Python client for the Khmer Name OCR Corrector API.
"""
import requests
import json
from typing import Dict, List, Optional, Any, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("khmer_client")

class KhmerNameCorrectorClient:
    """
    A client for interacting with the Khmer Name OCR Corrector API.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        
        Args:
            base_url: The base URL of the API
        """
        self.base_url = base_url.rstrip('/')
        logger.info(f"Initialized client for API at {self.base_url}")
    
    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """
        Make a request to the API.
        
        Args:
            method: HTTP method (get, post, etc.)
            endpoint: API endpoint
            data: Request data (for POST requests)
            
        Returns:
            Response data as a dictionary
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            if method.lower() == 'get':
                response = requests.get(url)
            elif method.lower() == 'post':
                response = requests.post(url, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {str(e)}")
            
            # Try to extract API error message if available
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    if 'detail' in error_data:
                        logger.error(f"API error detail: {error_data['detail']}")
                except:
                    pass
            
            raise
    
    def get_info(self) -> Dict:
        """
        Get information about the API and loaded model.
        
        Returns:
            Dictionary with API information
        """
        return self._request('get', '/info')
    
    def match(
        self,
        khmer_text: str,
        latin_text: Optional[str] = None,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        latin_weight: float = 0.5
    ) -> Dict:
        """
        Match a single Khmer name with optional Latin text.
        
        Args:
            khmer_text: Khmer text to match
            latin_text: Optional Latin/English text
            top_k: Number of results to return
            semantic_weight: Weight for semantic vs keyword matching
            latin_weight: Weight for Latin matching
            
        Returns:
            Dictionary with match results
        """
        data = {
            "khmer_text": khmer_text,
            "top_k": top_k,
            "semantic_weight": semantic_weight,
            "latin_weight": latin_weight
        }
        
        if latin_text:
            data["latin_text"] = latin_text
        
        return self._request('post', '/match', data)
    
    def batch_match(
        self,
        items: List[Dict[str, Any]]
    ) -> Dict:
        """
        Match multiple Khmer names in batch.
        
        Args:
            items: List of dictionaries with match parameters
            
        Returns:
            Dictionary with batch match results
        """
        # Validate and prepare the items
        for item in items:
            if 'khmer_text' not in item:
                raise ValueError("Each item must contain 'khmer_text'")
        
        data = {"items": items}
        return self._request('post', '/batch-match', data)
    
    def find_best_match(
        self,
        khmer_text: str,
        latin_text: Optional[str] = None,
        min_score: float = 0.5
    ) -> Optional[Dict]:
        """
        Find the best match for a Khmer name above a minimum score threshold.
        
        Args:
            khmer_text: Khmer text to match
            latin_text: Optional Latin/English text
            min_score: Minimum score threshold
            
        Returns:
            Dictionary with the best match or None if no match meets the threshold
        """
        response = self.match(khmer_text, latin_text)
        
        if not response.get('results'):
            return None
        
        # Find the best match
        best_match = response['results'][0]
        best_score = best_match['combined_score']
        
        if best_score >= min_score:
            return best_match
        
        return None 