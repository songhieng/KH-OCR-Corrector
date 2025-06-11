import streamlit as st
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
from typing import List, Tuple, Dict
from fuzzywuzzy import fuzz

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

# ─── MATCHER CLASS ─────────────────────────────────────────────────────────────
class KhmerNameMatcher:
    def __init__(self, names_file: str, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name, device="cpu")
        self.names, self.latin_map = self._load_names(names_file)
        self._build_index()

    def _load_names(self, filename: str) -> Tuple[List[str], Dict[str, str]]:
        names = []
        latin_map = {}
        
        print(f"Loading names from: {filename}")
        
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
                        
                        # Debug first few entries
                        if line_num <= 5:
                            print(f"Line {line_num}: Mapped Khmer '{khmer}' -> Latin '{latin}'")
                else:
                    # No comma or only one part
                    name = line.strip()
                    names.append(name)
                    
                    # Debug first few entries
                    if line_num <= 5:
                        print(f"Line {line_num}: Added name without Latin: '{name}'")
        
        print(f"Loaded {len(names)} Khmer names and {len(latin_map)} Latin mappings")
        return names, latin_map

    def _build_index(self):
        # Move computation to GPU and keep it there during encoding
        embeddings = self.model.encode(
            self.names,
            convert_to_tensor=True,
            device='cpu',  # <-- this is crucial
            show_progress_bar=True
        )

        # Move to CPU only after encoding
        emb_np = embeddings.detach().cpu().numpy()

        # Normalize for cosine similarity
        faiss.normalize_L2(emb_np)
        
        # Create and populate FAISS index
        dim = emb_np.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb_np)

    def match_name(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        q = query.strip()
        if not q:
            return []

        # GPU encoding
        emb = self.model.encode(
            [q],
            convert_to_tensor=True,
            device='cpu'  # GPU here
        )

        # Convert to NumPy for FAISS (FAISS is still on CPU)
        emb_np = emb.detach().cpu().numpy()
        faiss.normalize_L2(emb_np)

        # FAISS search
        scores, idxs = self.index.search(emb_np, min(top_k, len(self.names)))
        return [(self.names[i], float(scores[0][j])) for j, i in enumerate(idxs[0])]

    def keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not query: return []
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
        sem = dict(self.match_name(query, top_k*2))
        kw  = dict(self.keyword_search(query, top_k*2))
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
        Two-stage matching: first semantic+keyword match on Khmer, then refine with Latin matching
        Returns: List of (khmer_name, semantic_score, latin_name, combined_score, latin_score)
        """
        # First stage: semantic+keyword match on Khmer names
        khmer_matches = self.hybrid_match(khmer_query, top_k*2, semantic_weight)
        
        # Debug print to check the latin query
        print(f"Latin query: '{latin_query}'")
        print(f"Has Latin names: {self.has_latin_names()}")
        print(f"Latin map size: {len(self.latin_map)}")
        
        # Print some sample entries from latin_map
        if self.latin_map:
            print("Sample Latin mappings:")
            sample_keys = list(self.latin_map.keys())[:3]
            for k in sample_keys:
                print(f"  {k} -> {self.latin_map[k]}")
        
        # If we have no Latin names or no Latin query, just return the Khmer matches
        if not latin_query or not self.has_latin_names():
            return [(name, score, self.get_latin_name(name), score, 0.0) 
                   for name, score in khmer_matches[:top_k]]
        
        # Second stage: refine with Latin matching
        refined_matches = []
        for khmer_name, khmer_score in khmer_matches:
            latin_name = self.get_latin_name(khmer_name)
            print(f"For Khmer '{khmer_name}' found Latin '{latin_name}'")
            
            if latin_name:
                # Calculate fuzzy match score for Latin names (0-100)
                latin_score = 0.0
                
                # Direct string comparison first (case insensitive)
                if latin_query.lower() == latin_name.lower():
                    latin_score = 1.0
                    print(f"EXACT MATCH: '{latin_query}' == '{latin_name}'")
                else:
                    # Use FuzzyWuzzy if not exact match
                    fuzzy_score = fuzz.ratio(latin_query.lower(), latin_name.lower())
                    latin_score = fuzzy_score / 100.0
                    print(f"Fuzzy compare: '{latin_query}' vs '{latin_name}' -> {fuzzy_score}/100")
                
                # Combine scores with weighting
                combined_score = (khmer_score * (1-latin_weight)) + (latin_score * latin_weight)
                refined_matches.append((khmer_name, khmer_score, latin_name, combined_score, latin_score))
            else:
                # If no Latin name, just use the Khmer score
                refined_matches.append((khmer_name, khmer_score, "", khmer_score, 0.0))
        
        # Sort by combined score and return top-k
        refined_matches.sort(key=lambda x: x[3], reverse=True)
        return refined_matches[:top_k]

# ─── DATA LOADING & CACHE ──────────────────────────────────────────────────────
@st.cache_resource
def load_matcher(names_path: str):
    return KhmerNameMatcher(names_path)

# ─── UI LAYOUT ─────────────────────────────────────────────────────────────────
st.markdown("<div class='header'>Khmer Name OCR Corrector</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Upload a names file to begin. For best results, include Latin transliterations.</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded = st.file_uploader("Khmer Names File", type=["txt", "csv"], 
                               help="Format: 'khmer_name,latin_name' on each line for best results")
    if uploaded:
        names_path = "khmer_names_temp.txt"
        with open(names_path, 'wb') as f: f.write(uploaded.getvalue())
        matcher = load_matcher(names_path)
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
    semantic_weight = st.slider("Semantic vs. Keyword", 0.0, 1.0, 0.7,
                                help="High = more semantic, Low = more character-based")
    
    use_latin = st.checkbox("Use Latin/English matching", value=True)
    latin_weight = st.slider("Latin Match Weight", 0.0, 1.0, 0.5,
                           help="Weight of Latin name matching vs. Khmer matching")
    
    st.markdown("---")
    st.markdown("Need help? [GitHub Repo](#) | [Docs](#)")

# Main Input / Output
col1, col2 = st.columns((2,3), gap="large")
with col1:
    st.subheader("🔡 Enter OCR Text")
    if not matcher:
        st.info("Upload a names file in the sidebar to enable matching.")
    else:
        ocr_input_khmer = st.text_area("Paste Khmer OCR text (one name per line)", height=150,
                                 placeholder="e.g. លឹមហ៊ន")
        
        ocr_input_latin = st.text_area("Paste Latin/English OCR text (one name per line)", height=100,
                                     placeholder="e.g. Lim Hun", 
                                     disabled=not use_latin)
        
        if st.button("🔍 Match Names", type="primary"):
            if not ocr_input_khmer.strip():
                st.warning("Please enter some Khmer OCR text above.")
            else:
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
                        
                        # Show debug info above each match
                        if use_latin:
                            st.markdown(f"**Debug - Processing:**")
                            st.markdown(f"* Khmer input: `{khmer}`")
                            st.markdown(f"* Latin input: `{latin}`")
                        
                        if use_latin and matcher.has_latin_names() and latin:
                            st.markdown("Using Latin+Khmer matching")
                            results[khmer] = matcher.match_with_latin_fallback(
                                khmer, latin, top_k, semantic_weight, latin_weight
                            )
                        else:
                            if use_latin:
                                st.markdown("Falling back to Khmer-only matching")
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
                # Display the matched name with scores
                # Ensure no trailing commas in display names
                display_name = khmer_name.rstrip(',')
                
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