import chromadb
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image, ImageFilter
import os
import shutil
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FOLDER = os.path.join(BASE_DIR, "slide_memory_db")
IMAGES_FOLDER = os.path.join(BASE_DIR, "slide_images")
# Define where gold standard images actually live relative to this script
GOLD_FOLDER = os.path.join(IMAGES_FOLDER, "gold_standard")
TRAINING_FOLDER = os.path.join(IMAGES_FOLDER, "training_set")

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

# --- 2.5 CLASSIFICATION MODEL SETUP ---
# Custom Object for Model Loading
def grayscale_pipe(x):
    gray = tf.image.rgb_to_grayscale(x)
    return tf.image.grayscale_to_rgb(gray)

CLASSIFIER_MODEL_PATH = os.path.join(BASE_DIR, "multilabeling_model.keras")
classifier_model = None

CLASSIFIER_MODEL_PATH = os.path.join(BASE_DIR, "multilabeling_model.keras")
classifier_model = None

def load_local_classifier():
    """Loads and returns the classifier model so it can be cached externally."""
    try:
        if os.path.exists(CLASSIFIER_MODEL_PATH):
            print(f"Loading Classifier Model from {CLASSIFIER_MODEL_PATH}...")
            return load_model(CLASSIFIER_MODEL_PATH, custom_objects={"grayscale_pipe": grayscale_pipe})
        else:
            print("⚠️ Classifier model not found. Auto-tagging will be disabled.")
            return None
    except Exception as e:
        print(f"⚠️ Error loading classifier model: {e}")
        return None

# Threshold Configuration
THRESHOLD_MAP = {
    'Appendix_Reference':  0.3,
    'Data_Chart':          0.3, 
    'Framework_Structure': 0.7,  
    'Graphics_Visuals':    0.4,
    'Process_Flow':        0.25,  
    'Strategic_Text':      0.5,  
    'Title_Transition':    0.5
}
CLASS_NAMES = sorted(list(THRESHOLD_MAP.keys()))

# Load Existing Gold Standard Tags (The "Truth")
GOLD_TAGS_MAP = {}
TAGS_CSV_PATH = os.path.join(IMAGES_FOLDER, "training_set_tags.csv")
if os.path.exists(TAGS_CSV_PATH):
    try:
        df_tags = pd.read_csv(TAGS_CSV_PATH)
        # Assuming columns 'filename' and 'tags'
        if 'filename' in df_tags.columns and 'tags' in df_tags.columns:
            for idx, row in df_tags.iterrows():
                fname = str(row['filename']).strip()
                t_str = str(row['tags'])
                # Parse "Tag1, Tag2" -> ["Tag1", "Tag2"]
                t_list = [t.strip() for t in t_str.split(",") if t.strip()]
                GOLD_TAGS_MAP[fname] = t_list
        print(f"Loaded {len(GOLD_TAGS_MAP)} existing tag records.")
    except Exception as e:
        print(f"Error loading tags CSV: {e}")
else:
    print("⚠️ Tags CSV not found. Will rely on AI classification.")
    print("⚠️ Tags CSV not found. Will rely on AI classification.")

def append_tags_to_csv(filename, tags_list):
    """
    Appends a new record to the tags CSV. 
    Since the loader reads top-to-bottom, the last entry for a filename overrides previous ones.
    This acts as a persistent log.
    """
    if not tags_list: return
    
    try:
        # Format as CSV line: filename,"tag1, tag2"
        tags_str = ", ".join(tags_list)
        # Handle quotes if needed (simple approach)
        line = f'{filename},"{tags_str}"\n'
        
        with open(TAGS_CSV_PATH, "a") as f:
            f.write(line)
            
        # Update in-memory map immediately so we don't need to reload
        GOLD_TAGS_MAP[filename] = tags_list
        print(f"Persisted tags for {filename} to CSV.")
    except Exception as e:
        print(f"Failed to append tags to CSV: {e}")

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

# --- CLASSIFICATION HELPER ---
def classify_image(image_path):
    """
    Returns a list of predicted tags for a given image using the keras model.
    """
    """
    Returns a list of predicted tags for a given image using the keras model.
    """
    # Lazy load if not already loaded (allows for external caching or standalone use)
    global classifier_model
    if not classifier_model:
        classifier_model = load_local_classifier()
        
    if not classifier_model: return []
    
    try:
        # Load and Preprocess for MobileNetV2 (300x300)
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(300, 300))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0 # Rescale 1./255
        img_array = np.expand_dims(img_array, axis=0) # Batch dim
        
        # Predict
        probs = classifier_model.predict(img_array, verbose=0)[0]
        
        # Apply Thresholds
        predicted_tags = []
        for i, tag_name in enumerate(CLASS_NAMES):
            # Safe check index
            if i < len(probs):
                score = probs[i]
                threshold = THRESHOLD_MAP.get(tag_name, 0.5)
                if score > threshold:
                    predicted_tags.append(tag_name)
                    
        return predicted_tags
    except Exception as e:
        print(f"Classification error for {image_path}: {e}")
        return []

def get_tags_for_file(image_path, manual_tags=None):
    """
    Determines final tags for a file using the Hybrid Strategy:
    1. UI Manual Tags (Highest Priority)
    2. CSV Lookup (Existing Ground Truth)
    3. AI Model (Fallback/Auto)
    """
    filename = os.path.basename(image_path)
    
    # 1. Manual Override (UI)
    if manual_tags and len(manual_tags) > 0:
        return manual_tags
        
    # 2. CSV Lookup
    if filename in GOLD_TAGS_MAP:
        return GOLD_TAGS_MAP[filename]
        
    # 3. AI Classification
    return classify_image(image_path)


def add_slide_to_memory(image_path, text_content="", manual_tags=None):
    if not os.path.exists(image_path): return False
    
    # --- ENHANCED PERSISTENCE ---
    # 1. Ensure target directories exist
    os.makedirs(GOLD_FOLDER, exist_ok=True)
    os.makedirs(TRAINING_FOLDER, exist_ok=True)
    
    filename = os.path.basename(image_path)
    gold_path = os.path.join(GOLD_FOLDER, filename)
    training_path = os.path.join(TRAINING_FOLDER, filename)
    
    # 2. Copy file to Gold Standard and Training Set if not already there
    try:
        if not os.path.exists(gold_path):
            shutil.copy2(image_path, gold_path)
            print(f"Copied to Gold Standard: {gold_path}")
            
        if not os.path.exists(training_path):
            shutil.copy2(image_path, training_path)
            print(f"Copied to Training Set: {training_path}")
            
        # Update Stored Path to point to the Gold Standard version (Stable location)
        stored_path = os.path.relpath(gold_path, start=BASE_DIR)
        
    except Exception as e:
        print(f"Error copying files: {e}")
        # Fallback to original path if copy fails
        try:
            stored_path = os.path.relpath(image_path, start=BASE_DIR)
        except:
            stored_path = image_path

    slide_id = filename # Stick with filename as ID
    vis_emb = get_visual_embeddings(image_path)
    txt_emb = get_text_embeddings(text_content)
    
    if not vis_emb and not txt_emb: return False

    # Determine Tags
    final_tags = get_tags_for_file(image_path, manual_tags)
    
    # PERISTENCE: If these were manual tags, save them to CSV
    if manual_tags:
        append_tags_to_csv(os.path.basename(image_path), manual_tags)
    
    # Construct Metadata
    # Base metadata
    meta = {"source_path": stored_path}
    if text_content:
        meta["raw_text"] = text_content[:500]
        
    # Add Tag Flags (e.g. tag_Data_Chart = True)
    # Why flags? Easier for ChromaDB "where" filtering
    if final_tags:
        for tag in final_tags:
            meta[f"tag_{tag}"] = True
            
        # Also store as string list for display if finding generic way
        meta["all_tags"] = ", ".join(final_tags)

    try:
        if vis_emb:
            visual_collection.add(ids=[slide_id], embeddings=[vis_emb], metadatas=[meta])
        if txt_emb:
            text_collection.add(ids=[slide_id], embeddings=[txt_emb], metadatas=[meta])
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

def query_similar_slides(input_image_path, input_text_content="", n_results=3, visual_weight=0.7, filter_tags=None):
    """
    Returns a list of tuples: (path, method_used)
    method_used: "Filtered (Tag1, Tag2)" or "Fallback (Visual Match)"
    """
    if not os.path.exists(input_image_path): return []
    
    W_VISUAL = visual_weight
    W_TEXT = 1.0 - visual_weight
    
    vis_query = get_visual_embeddings(input_image_path)
    txt_query = get_text_embeddings(input_text_content)
    
    if not txt_query:
        W_VISUAL = 1.0
        W_TEXT = 0.0
    
    scores = {}
    method_used = "Standard Search"
    
    # Construct Where Clause
    where_clause = None
    if filter_tags and len(filter_tags) > 0:
        # We want to match if the slide has ANY of the filter_tags
        # ChromaDB $or syntax: {"$or": [{"tag_A": True}, {"tag_B": True}]}
        conditions = [{f"tag_{t}": True} for t in filter_tags]
        if len(conditions) == 1:
            where_clause = conditions[0]
        else:
            where_clause = {"$or": conditions}
        method_used = f"Filtered by {', '.join(filter_tags)}"
    
    # VISUAL SEARCH
    if vis_query:
        try:
            v_results = visual_collection.query(
                query_embeddings=[vis_query], 
                n_results=20,
                where=where_clause
            )
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
            t_results = text_collection.query(
                query_embeddings=[txt_query], 
                n_results=20,
                where=where_clause
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

    # --- FALLBACK LOGIC ---
    # If filter was active but we got NO results (scores is empty), try again WITHOUT filter
    if not scores and filter_tags:
        print("⚠️ No results with tags. Retrying without filter...")
        return query_similar_slides(input_image_path, input_text_content, n_results, visual_weight, filter_tags=None)
            
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
                # RETURN TUPLE: (path, method)
                final_paths.append((clean_path, method_used))
        except:
            pass
            
    return final_paths