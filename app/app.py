"""
Streamlit application for Khmer Name OCR Corrector.
"""
import os
import streamlit as st
import sys

# Add parent directory to path to allow importing from other modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.matcher import KhmerNameMatcher
from model.embedding import get_device, get_available_models

# ─── GLOBAL STYLES ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Khmer Name OCR Matcher", layout="wide", page_icon="")
st.markdown(
    """
    <style>
    .header {text-align: center; font-size: 2.5rem; font-weight: bold; margin-bottom: 1rem;}
    .subheader {font-size: 1.2rem; color: #555;}
    .card {background: #f9f9f9; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;}
    .card-title {font-weight: bold; font-size: 1.1rem; color: #000;}
    .score {float: right; font-size: 0.9rem; color: #777;}
    .latin {color: #555; font-size: 0.9rem; margin-top: 0.2rem;}
    .combined-score {color: #007bff; font-weight: bold;}
</style>
    """, unsafe_allow_html=True
)

# ─── DATA LOADING & CACHE ──────────────────────────────────────────────────────
@st.cache_resource
def load_matcher(names_path: str, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
    """Load and cache the name matcher model"""
    # Use model_dir if specified in environment
    model_dir = os.environ.get("MODEL_DIR")
    vector_db_dir = os.environ.get("VECTOR_DB_DIR")
    
    # Determine vector DB path if applicable
    vector_db_path = None
    if vector_db_dir:
        base_name = os.path.basename(names_path).split('.')[0]
        vector_db_path = os.path.join(vector_db_dir, f"{base_name}_{model_name.replace('/', '_')}.pkl")
    
    # Get best device
    device = get_device()
    
    return KhmerNameMatcher(
        names_file=names_path,
        model_name=model_name,
        device=device,
        vector_db_path=vector_db_path
    )

# ─── UI LAYOUT ─────────────────────────────────────────────────────────────────
st.markdown("<div class='header'>Khmer Name OCR Corrector</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Upload a names file to begin. For best results, include Latin transliterations.</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    available_models = [m["name"] for m in get_available_models()]
    default_model = "paraphrase-multilingual-mpnet-base-v2"
    selected_model = st.selectbox(
        "Embedding Model", 
        available_models,
        index=available_models.index(default_model) if default_model in available_models else 0,
        help="Select the embedding model to use for semantic matching"
    )
    
    # Names file upload
    uploaded = st.file_uploader(
        "Khmer Names File", 
        type=["txt", "csv"], 
        help="Format: 'latin_name,khmer_name' on each line for best results"
    )
    
    if uploaded:
        names_path = os.path.join("data", "temp", "khmer_names_temp.txt")
        os.makedirs(os.path.dirname(names_path), exist_ok=True)
        
        with open(names_path, 'wb') as f:
            f.write(uploaded.getvalue())
            
        with st.spinner("Loading matcher..."):
            matcher = load_matcher(names_path, selected_model)
            
        st.success(f"✅ Loaded **{len(matcher.names)}** names")
        if matcher.has_latin_names():
            st.success(f"✅ Latin transliterations available")
        else:
            st.warning("No Latin transliterations found. Add them for better results.")
    else:
        matcher = None
        st.warning("Please upload a names file.")

    st.subheader("Match Settings")
    top_k = st.slider("Top K Results", 1, 10, 5)
    semantic_weight = st.slider(
        "Semantic vs. Keyword", 
        0.0, 1.0, 0.7,
        help="High = more semantic, Low = more character-based"
    )
    
    use_latin = st.checkbox("Use Latin/English matching", value=True)
    latin_weight = st.slider(
        "Latin Match Weight", 
        0.0, 1.0, 0.5,
        help="Weight of Latin name matching vs. Khmer matching"
    )
    
    st.markdown("---")
    st.markdown("Khmer Name OCR Corrector - MLOps")

# Main Input / Output
col1, col2 = st.columns((2,3), gap="large")
with col1:
    st.subheader("🔡 Enter OCR Text")
    if not matcher:
        st.info("Upload a names file in the sidebar to enable matching.")
    else:
        ocr_input_khmer = st.text_area(
            "Paste Khmer OCR text (one name per line)", 
            height=150,
            placeholder="e.g. សុងហៀង"
        )
        
        ocr_input_latin = st.text_area(
            "Paste Latin/English OCR text (one name per line)", 
            height=100,
            placeholder="e.g. Songhieng", 
            disabled=not use_latin
        )
        
        if st.button("🔍 Match Names", type="primary"):
            if not ocr_input_khmer.strip():
                st.warning("Please enter some Khmer OCR text above.")
            else:
                with st.spinner("Matching names..."):
                    khmer_lines = [l for l in ocr_input_khmer.splitlines() if l.strip()]
                    latin_lines = [l for l in ocr_input_latin.splitlines() if l.strip()] if use_latin else []
                    
                    # Ensure same number of lines for both inputs if Latin is used
                    if use_latin and len(latin_lines) != len(khmer_lines):
                        st.error(f"Number of Latin names ({len(latin_lines)}) doesn't match Khmer names ({len(khmer_lines)})")
                    else:
                        # Match each name
                        results = {}
                        for i, khmer in enumerate(khmer_lines):
                            # Get corresponding Latin name if available
                            latin = ""
                            if use_latin and i < len(latin_lines):
                                latin = latin_lines[i].strip()
                            
                            if use_latin and matcher.has_latin_names() and latin:
                                results[khmer] = matcher.match_with_latin_fallback(
                                    khmer, latin, top_k, semantic_weight, latin_weight
                                )
                            else:
                                khmer_matches = matcher.hybrid_match(khmer, top_k, semantic_weight)
                                results[khmer] = [
                                    (name, score, matcher.get_latin_name(name), score, 0.0) 
                                    for name, score in khmer_matches
                                ]
                        st.session_state["results"] = results
                        st.session_state["use_latin"] = use_latin

with col2:
    st.subheader("📊 Results")
    if "results" in st.session_state:
        for line, matches in st.session_state["results"].items():
            st.markdown(f"**OCR Input:** `{line}`")
            for khmer_name, khmer_score, latin_name, combined_score, latin_score in matches:
                # Ensure no trailing commas in display names
                display_name = khmer_name.rstrip(',')
                
                # Display the matched name with scores
                if st.session_state.get("use_latin") and latin_name:
                    st.markdown(
                        f"<div class='card'>"
                        f"<span class='card-title'>{display_name}</span>"
                        f"<span class='score'>Khmer: {khmer_score:.3f}</span><br>"
                        f"<span class='score'>Latin: {latin_score:.3f}</span><br>"
                        f"<span class='combined-score'>Combined: {combined_score:.3f}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    # Create score displays without the Latin name
                    latin_score_html = ""
                    if latin_score > 0:
                        latin_score_html = f"<span class='score'>Latin: {latin_score:.3f}</span><br>"
                    
                    st.markdown(
                        f"<div class='card'>"
                        f"<span class='card-title'>{display_name}</span>"
                        f"<span class='score'>Khmer: {khmer_score:.3f}</span><br>"
                        f"{latin_score_html}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
    else:
        st.info("Your matches will appear here after uploading a names file and running a match.")

# Footer
st.markdown("---")
st.markdown("") 