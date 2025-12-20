import os
import google.generativeai as genai
from VB_encoder import query_similar_slides
import json
import logging
import streamlit as st

# --- 1. SILENCE WARNINGS (The "Slow Processor" Fix) ---
# This tells the system to ignore the specific "use_fast" warning from transformers
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 2. SET UP PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SECRETS_PATH = os.path.join(BASE_DIR, "secrets.json")

# --- 3. GET API KEY ---
# Getting Google API Key from .json file
# Getting Google API Key from .json file
GOOGLE_API_KEY = None

try:
    if "GOOGLE_API_KEY" in st.secrets:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

if not GOOGLE_API_KEY:
    if os.path.exists(SECRETS_PATH):
        try:
            with open(SECRETS_PATH, "r") as f:
                secrets = json.load(f)
                GOOGLE_API_KEY = secrets.get("GOOGLE_API_KEY")
        except:
            pass

if not GOOGLE_API_KEY:
    st.error("No API Key found! Check secrets.json locally or Secrets settings on Cloud.")
    st.stop() # Stops the app safely


# --- 4. CONFIGURE MODEL ---
# Configure the Generative Model

genai.configure(api_key=GOOGLE_API_KEY)

# DYNAMIC MODEL SELECTION
try:
    available_models = [m.name for m in genai.list_models()]
    
    # Priority list: Try to find the newest/best models first
    priority_keywords = [
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",  
        "gemini-2.5-flash",        
        "gemini-2.0-flash-lite",   
        "gemini-2.0-flash",
        "gemini-1.5-flash"
    ]
    
    selected_model_name = None
    
    for keyword in priority_keywords:
        match = next((m for m in available_models if keyword in m), None)
        if match:
            selected_model_name = match
            break

    if not selected_model_name:
        print("Error: No Gemini models found in your account.")
        exit()

    print(f"Using Model: {selected_model_name}")
    model = genai.GenerativeModel(selected_model_name)

except Exception as e:
    print(f"Error selecting model: {e}")
    exit()

# CRITIQUE SLIDE
def critique_slide(user_slide_path, user_slide_text="", visual_weight=0.7):
    
    # Get context for model slides
    print(f"Retrieving reference slides for: {os.path.basename(user_slide_path)}...")
    similar_paths = query_similar_slides(user_slide_path, input_text_content=user_slide_text, n_results=3, visual_weight=visual_weight)
    
    if not similar_paths:
        print("No similar slides found in memory.")
        return "Error: No context found."

    # Prepare inputs

    # Load the images as file data to send to the model
    def upload_to_gemini(path, mime_type="image/png"):
        file = genai.upload_file(path, mime_type=mime_type)
        return file
    
    # Upload User Slide
    user_file = upload_to_gemini(user_slide_path)
    
    # Upload Reference Slides
    reference_files = []
    for path, match_method in similar_paths:
        reference_files.append(upload_to_gemini(path))
    
    # C. THE PROMPT
    prompt_parts = [
        "Role: You are an expert presentation analyst.",
        "Task: Your task is to provide constructive feedback on an input slide by comparing it to three 'best-in-class' example presentations.",
        
        "\nInputs",
        "Best-in-Class Example 1: ", reference_files[0] if len(reference_files) > 0 else "N/A",
        "\nBest-in-Class Example 2: ", reference_files[1] if len(reference_files) > 1 else "N/A",
        "\nBest-in-Class Example 3: ", reference_files[2] if len(reference_files) > 2 else "N/A",
        
        "\nInput Slide to Analyze: ", user_file,

        "\nInstructions",
        "Analyze the three best-in-class examples to identify common principles of effective visual design and content structure.",
        "Evaluate the input slide using these identified principles as your criteria.",

        "\nOutput Format and Constraints",
        "Structure: Exactly two sections using Markdown H2 titles: ## Visual Feedback and ## Content Feedback.",
        "Quantity: Exactly three bullet points per section.",
        "Style: Each bullet point must be concise and start with a key takeaway in bold.",
        "Negative Constraint: Do not output anything but the requested feedback."
    ]

    # THE API CALL
    response = model.generate_content(prompt_parts)

    # Cleanup
    genai.delete_file(user_file.name)
    for f in reference_files: genai.delete_file(f.name)

    return response.text

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Check for images
    img_folder = os.path.join(os.path.dirname(__file__), "slide_images", "training_set")
    if not os.path.exists(img_folder):
        print("No images folder found.")
        exit()

    # Find the first .png file
    files = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith(".png")]
    
    if files:
        test_slide = files[3]
        print(f"Critiquing: {os.path.basename(test_slide)}")
        
        try:
            feedback = critique_slide(test_slide)
            print("\n" + "="*40)
            print("GEMINI CRITIQUE")
            print("="*40 + "\n")
            print(feedback)
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("No images found to test.")