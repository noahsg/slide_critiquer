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
    for path in similar_paths:
        reference_files.append(upload_to_gemini(path))
    
    # C. THE PROMPT
    prompt_parts = [
        "You are a Senior Principal at a top-tier consulting firm (MBB).",
        "Your task is to ruthlessly critique the 'User Draft Slide' against the high standards of the provided 'Reference Examples'.",
        
        "Below is the USER DRAFT SLIDE:",
        user_file,
        
        "Below are REFERENCE EXAMPLES (The Gold Standard):",
        *reference_files,
        
        "*** STYLE GUIDE (MIMIC THIS EXACT FORMAT AND TONE) ***",
        "Here is an example of the required output style. Notice the specific, imperative verbs and the split between Visual and Content:",
        
        """
        ### Visual Feedback
        - Change the font for text "this shows that we are...." to black for increased visibility
        - Realign the boxes on the left hand side to better improve aesthetic
        - Try using a pie chart to display percentage data versus a bar chart for better readability
        
        ### Content Feedback
        - Change the title to "Company X should do ...." so that it is more impactful and more clear
        - Add evidence to support your assumption that "Click through rate will increase"
        - Change the icons to better align with their corresponding text
        """,

        "*** INSTRUCTIONS FOR CURRENT SLIDE ***",
        "1. Analyze the User Draft. Identify the top 3 most critical Visual issues and top 3 Content issues.",
        "2. Do NOT copy the example text. Write NEW bullets specific to the User Draft.",
        "3. Use the exact format above: two headers (Visual Feedback, Content Feedback) with 3 bullets each.",
        "4. Start every bullet with a strong imperative verb (e.g., 'Remove', 'Align', 'Quantify').",
        "5. Be fluid: If the issue is whitespace, talk about whitespace. If it's data sorting, talk about data sorting.",

        "***You are limited to only outputing Visual Feedback and Content Feedback. Do not output any other text.***"
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
    img_folder = os.path.join(os.path.dirname(__file__), "slide_images")
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