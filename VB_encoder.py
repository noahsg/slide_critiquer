import chromadb
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image, ImageFilter
import os
import shutil

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FOLDER = os.path.join(BASE_DIR, "slide_memory_db")
IMAGES_FOLDER = os.path.join(BASE_DIR, "slide_images")

# --- 1. HARDWARE ACCELERATION ---
# Check for Mac (MPS) or Nvidia (CUDA), fallback to CPU
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("🚀 Using Apple Metal (MPS) acceleration!")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print("🚀 Using CUDA acceleration!")
else:
    DEVICE = "cpu"
    print("⚠️ Using CPU (Slower).")

# --- 2. MODEL SETUP ---
# --- 2. MODEL SETUP ---
# Visual Model (CLIP)
VISUAL_MODEL_NAME = "openai/clip-vit-base-patch32"
print(f"Loading Visual Model: {VISUAL_MODEL_NAME} to {DEVICE}...")
visual_processor = CLIPProcessor.from_pretrained(VISUAL_MODEL_NAME)
visual_model = CLIPModel.from_pretrained(VISUAL_MODEL_NAME).to(DEVICE)

# Text Model (MiniLM)
from sentence_transformers import SentenceTransformer
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
print(f"Loading Text Model: {TEXT_MODEL_NAME}...")
text_model = SentenceTransformer(TEXT_MODEL_NAME, device=DEVICE)

# --- 3. DATABASE SETUP ---
# We now maintain TWO collections: one for Visual, one for Text
chroma_client = chromadb.PersistentClient(path=DB_FOLDER)

# Visual Collection (Same as before)
visual_collection = chroma_client.get_or_create_collection(
    name="presentation_slides",
    metadata={"hnsw:space": "cosine"}
)

# Text Collection (New)
text_collection = chroma_client.get_or_create_collection(
    name="presentation_text",
    metadata={"hnsw:space": "cosine"}
)

def force_reinit():
    """Forcibly re-initializes the ChromaDB client and collections."""
    global visual_collection, text_collection, chroma_client
    try:
        print("🔄 Forcing re-initialization of DB connection...")
        chroma_client = chromadb.PersistentClient(path=DB_FOLDER)
        
        visual_collection = chroma_client.get_or_create_collection(
            name="presentation_slides",
            metadata={"hnsw:space": "cosine"}
        )
        
        text_collection = chroma_client.get_or_create_collection(
            name="presentation_text",
            metadata={"hnsw:space": "cosine"}
        )
        return True
    except Exception as e:
        print(f"Failed to reinit: {e}")
        return False

def reset_memory():
    """Wipes the database clean."""
    global visual_collection, text_collection, chroma_client
    try:
        try:
            chroma_client.delete_collection("presentation_slides")
            chroma_client.delete_collection("presentation_text")
        except Exception as e:
            print(f"Collection delete warning: {e}")
            pass
            
        # Recreate immediately
        force_reinit()
        print("Memory reset successfully (Visual + Text).")
        return True
    except Exception as e:
        print(f"Error resetting memory: {e}")
        return False

def process_image_for_visual_model(image_path):
    """Helper: Opens image and converts to RGB (No Wireframe)."""
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Grayscale
        gray = image.convert("L")
        
        # KEY CHANGE: MinFilter (Erosion) effectively merges text characters into blocks
        # and thickens thin structural lines, creating a robust "Layout Map".
        dilated = gray.filter(ImageFilter.MinFilter(size=5))
        
        # Edge Detection
        edges = dilated.filter(ImageFilter.FIND_EDGES)
        return edges.convert("RGB")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def get_visual_embeddings(image_path):
    """Generates embedding for a single image using CLIP."""
    img = process_image_for_visual_model(image_path)
    if not img: return None
    
    try:
        inputs = visual_processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            features = visual_model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            
        return features.cpu().tolist()[0]
    except Exception as e:
        print(f"Visual encoding error: {e}")
        return None

def get_text_embeddings(text):
    """Generates embedding for text using MiniLM."""
    if not text or len(text.strip()) == 0:
        return None
    try:
        # SentenceTransformer handles tokenization and device placement
        embedding = text_model.encode(text, convert_to_tensor=True, device=DEVICE)
        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
        return embedding.cpu().tolist()
    except Exception as e:
        print(f"Text encoding error: {e}")
        return None

def add_slide_to_memory(image_path, text_content=""):
    """
    Adds a slide to BOTH Visual and Text memory.
    The ID is the image filename for both.
    """
    if not os.path.exists(image_path): return False
    
    slide_id = os.path.basename(image_path)
    
    # 1. Visual Embedding
    vis_emb = get_visual_embeddings(image_path)
    
    # 2. Text Embedding
    txt_emb = get_text_embeddings(text_content)
    
    if not vis_emb and not txt_emb:
        print(f"Skipping {slide_id}: No visual OR text content.")
        return False

    try:
        # Add to Visual Collection
        if vis_emb:
            visual_collection.add(
                ids=[slide_id],
                embeddings=[vis_emb],
                metadatas=[{"source_path": image_path}]
            )
            
        # Add to Text Collection (only if text exists)
        if txt_emb:
            text_collection.add(
                ids=[slide_id],
                embeddings=[txt_emb],
                metadatas=[{"source_path": image_path, "raw_text": text_content[:500]}] # Store snippet
            )
            
        return True
    except Exception as e:
        print(f"Error adding slide {slide_id}: {e}")
        return False

def query_similar_slides(input_image_path, input_text_content="", n_results=3, visual_weight=0.7):
    """
    Hybrid Search with dynamic weighting.
    visual_weight: Float 0.0 to 1.0 (Text weight = 1.0 - visual_weight)
    """
    if not os.path.exists(input_image_path): return []
    
    # Weights
    W_VISUAL = visual_weight
    W_TEXT = 1.0 - visual_weight
    
    # 1. Get Query Vectors
    vis_query = get_visual_embeddings(input_image_path)
    txt_query = get_text_embeddings(input_text_content)
    
    # If no text provided, we still want to try to extract it from the image path if possible?
    # For now, we assume the caller provides text if they want text search.
    # If text is missing, we fallback to 100% visual.
    if not txt_query:
        W_VISUAL = 1.0
        W_TEXT = 0.0
    
    # Local scoring dict: {slide_id: final_score}
    scores = {}
    
    # --- VISUAL SEARCH ---
    if vis_query:
        try:
            # We ask for more results (e.g. 20) to have a candidate pool for fusion
            v_results = visual_collection.query(
                query_embeddings=[vis_query],
                n_results=20 
            )
            
            # Chroma returns distances (smaller is better). We need similarity (0-1).
            # Approx similarity = 1 - distance (for cosine distance)
            if v_results['ids'] and len(v_results['ids'][0]) > 0:
                ids = v_results['ids'][0]
                dists = v_results['distances'][0]
                metas = v_results['metadatas'][0]
                
                for i, sid in enumerate(ids):
                    # Convert distance to similarity score
                    sim = max(0, 1.0 - dists[i])
                    
                    if sid not in scores: scores[sid] = 0.0
                    scores[sid] += (sim * W_VISUAL)
                    
                    # Store path for retrieval
                    # We hide it in the key's object or a separate map
                    # Since we return paths, let's map ID -> Path
        except Exception as e:
            print(f"Visual search failed: {e}")

    # --- TEXT SEARCH ---
    if txt_query:
        try:
            t_results = text_collection.query(
                query_embeddings=[txt_query],
                n_results=20
            )
            
            if t_results['ids'] and len(t_results['ids'][0]) > 0:
                ids = t_results['ids'][0]
                dists = t_results['distances'][0]
                
                for i, sid in enumerate(ids):
                    sim = max(0, 1.0 - dists[i])
                    
                    if sid not in scores: scores[sid] = 0.0
                    scores[sid] += (sim * W_TEXT)
        except Exception as e:
            print(f"Text search failed: {e}")
            
    # Sort by score DESC
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return Top N paths
    final_paths = []
    
    # We need to resolve ID -> Path.
    # The 'metadatas' from visual search usually has it. 
    # But if a result appears ONLY in text search, we need its path.
    # Strategy: One quick lookup or carry it.
    # Simpler: We'll assume everything in Text is in Visual or we can query DB for path.
    # For speed, let's just grab the path if we have it or reconstruc it.
    # Actually, we stored 'source_path' in metadata for both.
    # We can fetch it.
    
    for sid, score in sorted_items[:n_results]:
        # We need the path.
        # Check standard folder structure? Or query DB?
        # Querying DB by ID is safer.
        
        try:
            # Try visual collection first
            r = visual_collection.get(ids=[sid])
            if r['metadatas'] and len(r['metadatas']) > 0:
                final_paths.append(r['metadatas'][0]['source_path'])
            else:
                # Try text collection
                r2 = text_collection.get(ids=[sid])
                if r2['metadatas'] and len(r2['metadatas']) > 0:
                    final_paths.append(r2['metadatas'][0]['source_path'])
        except:
            pass
            
    return final_paths

if __name__ == "__main__":
    pass