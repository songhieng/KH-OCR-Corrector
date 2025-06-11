"""
FastAPI application for Khmer Name OCR Corrector.
"""
import os
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uvicorn

from utils.matcher import KhmerNameMatcher
from model.embedding import load_model, get_device

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api")

# Create FastAPI app
app = FastAPI(
    title="Khmer Name OCR Corrector API",
    description="API for matching and correcting OCR-processed Khmer names",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global matcher instance
matcher = None

# Pydantic models for request/response
class MatchRequest(BaseModel):
    khmer_text: str = Field(..., description="Khmer text from OCR")
    latin_text: Optional[str] = Field(None, description="Latin/English text from OCR (optional)")
    top_k: int = Field(5, description="Number of results to return")
    semantic_weight: float = Field(0.7, description="Weight for semantic vs keyword matching")
    latin_weight: float = Field(0.5, description="Weight for Latin matching")

class MatchResult(BaseModel):
    khmer_name: str = Field(..., description="Matched Khmer name")
    khmer_score: float = Field(..., description="Semantic matching score")
    latin_name: Optional[str] = Field(None, description="Latin/English transliteration")
    latin_score: float = Field(0.0, description="Latin matching score")
    combined_score: float = Field(..., description="Combined score")

class MatchResponse(BaseModel):
    results: List[MatchResult] = Field(..., description="Match results")
    input_khmer: str = Field(..., description="Input Khmer text")
    input_latin: Optional[str] = Field(None, description="Input Latin text")

class BatchMatchRequest(BaseModel):
    items: List[MatchRequest] = Field(..., description="Batch of match requests")

class BatchMatchResponse(BaseModel):
    results: List[MatchResponse] = Field(..., description="Batch match results")

class ModelInfo(BaseModel):
    name: str = Field(..., description="Model name")
    device: str = Field(..., description="Device used for inference")
    vector_db: Optional[str] = Field(None, description="Vector database path if used")
    has_latin_names: bool = Field(False, description="Whether Latin names are available")
    num_names: int = Field(0, description="Number of names in the database")

def get_matcher():
    """
    Get or initialize the global matcher instance.
    """
    global matcher
    if matcher is None:
        # Get configuration from environment variables
        model_name = os.environ.get("MODEL_NAME", "paraphrase-multilingual-mpnet-base-v2")
        names_file = os.environ.get("NAMES_FILE", None)
        vector_db_path = os.environ.get("VECTOR_DB_PATH", None)
        model_dir = os.environ.get("MODEL_DIR", "model")
        device = os.environ.get("DEVICE", get_device())
        
        if not names_file and not vector_db_path:
            raise ValueError("Either NAMES_FILE or VECTOR_DB_PATH must be provided")
        
        logger.info(f"Initializing matcher with model={model_name}, device={device}")
        matcher = KhmerNameMatcher(
            names_file=names_file,
            model_name=model_name,
            device=device,
            vector_db_path=vector_db_path
        )
        logger.info(f"Matcher initialized with {len(matcher.names)} names")
    
    return matcher

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint that returns API information."""
    return {
        "name": "Khmer Name OCR Corrector API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/info", response_model=ModelInfo)
async def info(matcher: KhmerNameMatcher = Depends(get_matcher)):
    """Get information about the loaded model and data."""
    return ModelInfo(
        name=matcher.model_name,
        device=matcher.device,
        vector_db=matcher.vector_db_path,
        has_latin_names=matcher.has_latin_names(),
        num_names=len(matcher.names)
    )

@app.post("/match", response_model=MatchResponse)
async def match(request: MatchRequest, matcher: KhmerNameMatcher = Depends(get_matcher)):
    """
    Match a single Khmer name (with optional Latin transliteration).
    """
    try:
        if not request.khmer_text.strip():
            raise HTTPException(status_code=400, detail="Khmer text is required")
        
        # Perform matching
        if request.latin_text and matcher.has_latin_names():
            matches = matcher.match_with_latin_fallback(
                request.khmer_text,
                request.latin_text,
                request.top_k,
                request.semantic_weight,
                request.latin_weight
            )
            results = [
                MatchResult(
                    khmer_name=khmer_name,
                    khmer_score=khmer_score,
                    latin_name=latin_name,
                    latin_score=latin_score,
                    combined_score=combined_score
                )
                for khmer_name, khmer_score, latin_name, combined_score, latin_score in matches
            ]
        else:
            # Khmer-only matching
            khmer_matches = matcher.hybrid_match(
                request.khmer_text,
                request.top_k,
                request.semantic_weight
            )
            results = [
                MatchResult(
                    khmer_name=name,
                    khmer_score=score,
                    latin_name=matcher.get_latin_name(name),
                    latin_score=0.0,
                    combined_score=score
                )
                for name, score in khmer_matches
            ]
        
        return MatchResponse(
            results=results,
            input_khmer=request.khmer_text,
            input_latin=request.latin_text
        )
    
    except Exception as e:
        logger.error(f"Error in match endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-match", response_model=BatchMatchResponse)
async def batch_match(request: BatchMatchRequest, matcher: KhmerNameMatcher = Depends(get_matcher)):
    """
    Match a batch of Khmer names (with optional Latin transliterations).
    """
    try:
        responses = []
        for item in request.items:
            if not item.khmer_text.strip():
                continue
                
            # Perform matching
            if item.latin_text and matcher.has_latin_names():
                matches = matcher.match_with_latin_fallback(
                    item.khmer_text,
                    item.latin_text,
                    item.top_k,
                    item.semantic_weight,
                    item.latin_weight
                )
                results = [
                    MatchResult(
                        khmer_name=khmer_name,
                        khmer_score=khmer_score,
                        latin_name=latin_name,
                        latin_score=latin_score,
                        combined_score=combined_score
                    )
                    for khmer_name, khmer_score, latin_name, combined_score, latin_score in matches
                ]
            else:
                # Khmer-only matching
                khmer_matches = matcher.hybrid_match(
                    item.khmer_text,
                    item.top_k,
                    item.semantic_weight
                )
                results = [
                    MatchResult(
                        khmer_name=name,
                        khmer_score=score,
                        latin_name=matcher.get_latin_name(name),
                        latin_score=0.0,
                        combined_score=score
                    )
                    for name, score in khmer_matches
                ]
            
            responses.append(
                MatchResponse(
                    results=results,
                    input_khmer=item.khmer_text,
                    input_latin=item.latin_text
                )
            )
        
        return BatchMatchResponse(results=responses)
    
    except Exception as e:
        logger.error(f"Error in batch-match endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run with uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=True) 