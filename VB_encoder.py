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
# Define where gold standard images actually live relative to this script
GOLD_FOLDER = os.path.join(IMAGES_FOLDER, "gold_standard")

# --- 1. HARDWARE ACCELERATION ---
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
chroma_client = chromadb.PersistentClient(path=DB_FOLDER)

visual_collection = chroma_client.get_or_create_collection(
    name="presentation_slides",
    metadata={"hnsw:space": "cosine"}
)

text_collection = chroma_client.get_or_create_collection(
    name="presentation_text",
    metadata={"hnsw:space": "cosine"}
)

def force_reinit():
    global visual_collection, text_collection, chroma_client
    try:
        print("🔄 Forcing re-initialization of DB connection...")
        chroma_client = chromadb.PersistentClient(path=DB_FOLDER)
        visual_collection = chroma_client.get_or_create_collection(name="presentation_slides", metadata={"hnsw:space": "cosine"})
        text_collection = chroma_client.get_or_create_collection(name="presentation_text", metadata={"hnsw:space": "cosine"})
        return True
    except Exception as e:
        print(f"Failed to reinit: {e}")
        return False

def reset_memory():
    global visual_collection, text_collection, chroma_client
    try:
        try:
            chroma_client.delete_collection("presentation_slides")
            chroma_client.delete_collection("presentation_text")
        except Exception as e:
            pass
        force_reinit()
        return True
    except Exception as e:
        print(f"Error resetting memory: {e}")
        return False

def process_image_for_visual_model(image_path):
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        gray = image.convert("L")
        dilated = gray.filter(ImageFilter.MinFilter(size=5))
        edges = dilated.filter(ImageFilter.FIND_EDGES)
        return edges.convert("RGB")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def get_visual_embeddings(image_path):
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
    if not text or len(text.strip()) == 0: return None
    try:
        embedding = text_model.encode(text, convert_to_tensor=True, device=DEVICE)
        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
        return embedding.cpu().tolist()
    except Exception as e:
        print(f"Text encoding error: {e}")
        return None

def add_slide_to_memory(image_path, text_content=""):
    if not os.path.exists(image_path): return False
    
    # Store RELATIVE path to avoid "Users/Noah" issues in future
    try:
        stored_path = os.path.relpath(image_path, start=BASE_DIR)
    except:
        stored_path = image_path # Fallback

    slide_id = os.path.basename(image_path)
    vis_emb = get_visual_embeddings(image_path)
    txt_emb = get_text_embeddings(text_content)
    
    if not vis_emb and not txt_emb: return False

    try:
        if vis_emb:
            visual_collection.add(ids=[slide_id], embeddings=[vis_emb], metadatas=[{"source_path": stored_path}])
        if txt_emb:
            text_collection.add(ids=[slide_id], embeddings=[txt_emb], metadatas=[{"source_path": stored_path, "raw_text": text_content[:500]}])
        return True
    except Exception as e:
        print(f"Error adding slide {slide_id}: {e}")
        return False

def sanitize_path(retrieved_path):
    """
    Fixes paths that point to the wrong OS structure (e.g. Mac paths on Linux).
    """
    # 1. If path exists as-is, return it.
    if os.path.exists(retrieved_path):
        return retrieved_path

    # 2. Extract just the filename (e.g., "slide_1.png")
    filename = os.path.basename(retrieved_path)
    
    # 3. Construct the probable path in the current environment
    # We assume gold standard slides are in 'slide_images/gold_standard'
    likely_path = os.path.join(GOLD_FOLDER, filename)
    
    if os.path.exists(likely_path):
        return likely_path
        
    # 4. Fallback: check just the images folder
    fallback_path = os.path.join(IMAGES_FOLDER, filename)
    if os.path.exists(fallback_path):
        return fallback_path
        
    # 5. Return original if nothing works (will cause error, but we tried)
    return retrieved_path

def query_similar_slides(input_image_path, input_text_content="", n_results=3, visual_weight=0.7):
    if not os.path.exists(input_image_path): return []
    
    W_VISUAL = visual_weight
    W_TEXT = 1.0 - visual_weight
    
    vis_query = get_visual_embeddings(input_image_path)
    txt_query = get_text_embeddings(input_text_content)
    
    if not txt_query:
        W_VISUAL = 1.0
        W_TEXT = 0.0
    
    scores = {}
    
    # VISUAL SEARCH
    if vis_query:
        try:
            v_results = visual_collection.query(query_embeddings=[vis_query], n_results=20)
            if v_results['ids'] and len(v_results['ids'][0]) > 0:
                ids = v_results['ids'][0]
                dists = v_results['distances'][0]
                for i, sid in enumerate(ids):
                    sim = max(0, 1.0 - dists[i])
                    if sid not in scores: scores[sid] = 0.0
                    scores[sid] += (sim * W_VISUAL)
        except Exception as e:
            print(f"Visual search failed: {e}")

    # TEXT SEARCH
    if txt_query:
        try:
            t_results = text_collection.query(query_embeddings=[txt_query], n_results=20)
            if t_results['ids'] and len(t_results['ids'][0]) > 0:
                ids = t_results['ids'][0]
                dists = t_results['distances'][0]
                for i, sid in enumerate(ids):
                    sim = max(0, 1.0 - dists[i])
                    if sid not in scores: scores[sid] = 0.0
                    scores[sid] += (sim * W_TEXT)
        except Exception as e:
            print(f"Text search failed: {e}")
            
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    final_paths = []
    
    for sid, score in sorted_items[:n_results]:
        try:
            # Try getting path from visual collection
            r = visual_collection.get(ids=[sid])
            path_found = None
            if r['metadatas'] and len(r['metadatas']) > 0:
                path_found = r['metadatas'][0]['source_path']
            else:
                # Try text collection
                r2 = text_collection.get(ids=[sid])
                if r2['metadatas'] and len(r2['metadatas']) > 0:
                    path_found = r2['metadatas'][0]['source_path']
            
            # SANITIZE THE PATH
            if path_found:
                clean_path = sanitize_path(path_found)
                final_paths.append(clean_path)
        except:
            pass
            
    return final_paths