import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
from typing import List, Tuple, Dict, Optional
import os
import pickle
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("khmer_matcher")

class KhmerNameMatcher:
    """
    A class for matching Khmer names using both semantic search and Latin name matching.
    """
    
    def __init__(
        self, 
        names_file: Optional[str] = None,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        device: str = "cpu",
        vector_db_path: Optional[str] = None,
        precomputed_embeddings: Optional[np.ndarray] = None,
        names: Optional[List[str]] = None,
        latin_map: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the KhmerNameMatcher.
        
        Args:
            names_file: Path to a file containing names (and optionally Latin transliterations)
            model_name: Name of the sentence transformer model to use
            device: Device to use for model inference ('cpu' or 'cuda')
            vector_db_path: Path to save/load vector database
            precomputed_embeddings: Pre-computed embeddings (optional)
            names: List of names (optional, used with precomputed_embeddings)
            latin_map: Dictionary mapping Khmer names to Latin names (optional)
        """
        self.model_name = model_name
        self.device = device
        self.vector_db_path = vector_db_path
        
        # Load the model
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        
        # Initialize from provided data or file
        if precomputed_embeddings is not None and names is not None:
            logger.info("Using provided precomputed embeddings")
            self.embeddings = precomputed_embeddings
            self.names = names
            self.latin_map = latin_map or {}
            self._build_index()
        elif names_file:
            logger.info(f"Loading names from file: {names_file}")
            self.names, self.latin_map = self._load_names(names_file)
            
            # Try loading from vector DB if path is provided
            if vector_db_path and os.path.exists(vector_db_path):
                logger.info(f"Loading vector database from: {vector_db_path}")
                self._load_vector_db()
            else:
                logger.info("Building index from scratch")
                self._build_index()
                
                # Save to vector DB if path is provided
                if vector_db_path:
                    logger.info(f"Saving vector database to: {vector_db_path}")
                    self._save_vector_db()
        else:
            raise ValueError("Either names_file or (precomputed_embeddings and names) must be provided")

    def _load_names(self, filename: str) -> Tuple[List[str], Dict[str, str]]:
        """
        Load names from a file.
        
        The file can be in one of two formats:
        1. One name per line
        2. latin,khmer pairs (one per line)
        
        Returns:
            Tuple of (list of names, dict mapping from khmer to latin)
        """
        names = []
        latin_map = {}
        
        logger.info(f"Loading names from: {filename}")
        
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Check if the file contains Latin,Khmer pairs
                parts = line.split(',', 1)
                if len(parts) >= 2:
                    latin = parts[0].strip()
                    # Handle any trailing commas in the Khmer part
                    khmer = parts[1].strip().rstrip(',')
                    
                    if khmer and latin:
                        names.append(khmer)  # We store Khmer names in the index
                        latin_map[khmer] = latin  # Map from Khmer -> Latin
                else:
                    # No comma or only one part
                    name = line.strip()
                    names.append(name)
        
        logger.info(f"Loaded {len(names)} Khmer names and {len(latin_map)} Latin mappings")
        return names, latin_map

    def _build_index(self):
        """Build the FAISS index for fast similarity search"""
        # Compute embeddings
        logger.info(f"Computing embeddings for {len(self.names)} names")
        self.embeddings = self.model.encode(
            self.names,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=True
        )

        # Convert to numpy for FAISS
        emb_np = self.embeddings.detach().cpu().numpy()

        # Normalize for cosine similarity
        faiss.normalize_L2(emb_np)
        
        # Create and populate FAISS index
        dim = emb_np.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb_np)
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors of dimension {dim}")

    def _save_vector_db(self):
        """Save the vector database to disk"""
        if not self.vector_db_path:
            logger.warning("No vector_db_path provided, cannot save vector database")
            return
            
        os.makedirs(os.path.dirname(self.vector_db_path), exist_ok=True)
        
        # Save the index, names, and latin_map
        with open(self.vector_db_path, 'wb') as f:
            pickle.dump({
                'names': self.names,
                'latin_map': self.latin_map,
                'embeddings': self.embeddings.detach().cpu().numpy() if torch.is_tensor(self.embeddings) else self.embeddings,
                'model_name': self.model_name
            }, f)
        logger.info(f"Saved vector database to {self.vector_db_path}")

    def _load_vector_db(self):
        """Load the vector database from disk"""
        if not self.vector_db_path or not os.path.exists(self.vector_db_path):
            logger.warning(f"Vector database not found at {self.vector_db_path}")
            return False
            
        with open(self.vector_db_path, 'rb') as f:
            data = pickle.load(f)
            
        # Check if the model name matches
        if data.get('model_name') != self.model_name:
            logger.warning(f"Model mismatch: saved={data.get('model_name')}, current={self.model_name}")
            return False
            
        self.names = data['names']
        self.latin_map = data['latin_map']
        self.embeddings = data['embeddings']
        
        # Rebuild the index
        self._build_index()
        logger.info(f"Loaded vector database from {self.vector_db_path}")
        return True

    def match_name(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the top-k semantic matches for a query.
        
        Args:
            query: The query string to match
            top_k: Number of results to return
            
        Returns:
            List of (name, score) tuples
        """
        q = query.strip()
        if not q:
            return []

        # Encode the query
        emb = self.model.encode(
            [q],
            convert_to_tensor=True,
            device=self.device
        )

        # Convert to NumPy for FAISS
        emb_np = emb.detach().cpu().numpy()
        faiss.normalize_L2(emb_np)

        # FAISS search
        scores, idxs = self.index.search(emb_np, min(top_k, len(self.names)))
        return [(self.names[i], float(scores[0][j])) for j, i in enumerate(idxs[0])]

    def keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the top-k character-based matches for a query.
        
        Args:
            query: The query string to match
            top_k: Number of results to return
            
        Returns:
            List of (name, score) tuples
        """
        if not query: 
            return []
            
        scores = []
        qset = set(query)
        
        for name in self.names:
            nset = set(name)
            overlap = len(qset & nset)
            union = len(qset | nset) or 1
            scores.append((name, overlap/union))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def hybrid_match(self, query: str, top_k: int = 5, semantic_weight: float = 0.7) -> List[Tuple[str, float]]:
        """
        Combine semantic and keyword matching for better results.
        
        Args:
            query: The query string to match
            top_k: Number of results to return
            semantic_weight: Weight to give to semantic matching vs keyword matching
            
        Returns:
            List of (name, score) tuples
        """
        sem = dict(self.match_name(query, top_k*2))
        kw = dict(self.keyword_search(query, top_k*2))
        
        combined = {name: s * semantic_weight for name, s in sem.items()}
        for name, k in kw.items():
            combined[name] = combined.get(name, 0) + k * (1 - semantic_weight)
            
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
    def has_latin_names(self) -> bool:
        """Check if the loaded dataset has Latin name mappings"""
        return len(self.latin_map) > 0
    
    def get_latin_name(self, khmer_name: str) -> str:
        """Get the Latin transliteration for a Khmer name if available"""
        return self.latin_map.get(khmer_name, "")
    
    def match_with_latin_fallback(self, khmer_query: str, latin_query: str, 
                                 top_k: int = 5, semantic_weight: float = 0.7,
                                 latin_weight: float = 0.5) -> List[Tuple[str, float, str, float, float]]:
        """
        Two-stage matching: first semantic+keyword match on Khmer, then refine with Latin matching.
        
        Args:
            khmer_query: Khmer text to match
            latin_query: Latin/English text to match
            top_k: Number of results to return
            semantic_weight: Weight for semantic vs keyword matching in first stage
            latin_weight: Weight for Latin matching in second stage
            
        Returns:
            List of (khmer_name, semantic_score, latin_name, combined_score, latin_score) tuples
        """
        # First stage: semantic+keyword match on Khmer names
        khmer_matches = self.hybrid_match(khmer_query, top_k*2, semantic_weight)
        
        # If we have no Latin names or no Latin query, just return the Khmer matches
        if not latin_query or not self.has_latin_names():
            return [(name, score, self.get_latin_name(name), score, 0.0) 
                   for name, score in khmer_matches[:top_k]]
        
        # Second stage: refine with Latin matching
        refined_matches = []
        for khmer_name, khmer_score in khmer_matches:
            latin_name = self.get_latin_name(khmer_name)
            
            if latin_name:
                # Calculate fuzzy match score for Latin names
                latin_score = 0.0
                
                # Direct string comparison first (case insensitive)
                if latin_query.lower() == latin_name.lower():
                    latin_score = 1.0
                else:
                    # Use FuzzyWuzzy if not exact match
                    fuzzy_score = fuzz.ratio(latin_query.lower(), latin_name.lower())
                    latin_score = fuzzy_score / 100.0
                
                # Combine scores with weighting
                combined_score = (khmer_score * (1-latin_weight)) + (latin_score * latin_weight)
                refined_matches.append((khmer_name, khmer_score, latin_name, combined_score, latin_score))
            else:
                # If no Latin name, just use the Khmer score
                refined_matches.append((khmer_name, khmer_score, "", khmer_score, 0.0))
        
        # Sort by combined score and return top-k
        refined_matches.sort(key=lambda x: x[3], reverse=True)
        return refined_matches[:top_k] 