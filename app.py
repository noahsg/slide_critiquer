import streamlit as st
import os
import shutil
from pathlib import Path
import logging

# Import backend logic
# Ensure these files are in the same directory or accessible via pythonpath
import Slide_Critic
import VB_encoder

import PDF_Translator
import Cloud_Converter
import fitz # PyMuPDF
import time

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_IMG_DIR = os.path.join(BASE_DIR, "slide_images", "user_temp")
GOLD_IMG_DIR = os.path.join(BASE_DIR, "slide_images", "gold_standard")
TEMP_UPLOADS_DIR = "temp_uploads"

# Ensure directories exist
os.makedirs(USER_IMG_DIR, exist_ok=True)
os.makedirs(GOLD_IMG_DIR, exist_ok=True)
os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Consulting Slide Critique Tool",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- STYLING (CSS) ---
# Inject custom CSS to match the requested "clean corporate" aesthetic
st.markdown("""
<style>
    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-weight: 600;
        color: #0F172A;
    }
    
    /* Buttons - Primary */
    div.stButton > button {
        background-color: #0F172A;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        width: 100%;
        font-weight: 500;
    }
    div.stButton > button:hover {
        background-color: #1E293B;
        color: white;
        border-color: #1E293B;
    }

    /* Container Styling (White Cards) */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .st-emotion-cache-16idsys p {
        font-size: 16px;
    }

    /* Upload Box Styling */
    [data-testid='stFileUploader'] {
        border: 1px dashed #CBD5E1;
        border-radius: 8px;
        padding: 2rem;
        background-color: #F8FAFC;
    }

    /* Success Message */
    .stSuccess {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #6EE7B7;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# --- UTILITY FUNCTIONS ---

def save_uploaded_file(uploaded_file, directory=TEMP_UPLOADS_DIR):
    """Saves a Streamlit uploaded file to a temporary directory."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def clear_workspace():
    """Deletes temporary user files."""
    try:
        # Clear User Images
        if os.path.exists(USER_IMG_DIR):
            shutil.rmtree(USER_IMG_DIR)
            os.makedirs(USER_IMG_DIR) # Recreate empty folder
            
        # Clear Temp Uploads
        if os.path.exists(TEMP_UPLOADS_DIR):
            shutil.rmtree(TEMP_UPLOADS_DIR)
            os.makedirs(TEMP_UPLOADS_DIR)

        return True
    except Exception as e:
        print(f"Error clearing workspace: {e}")
        return False

# --- APP LAYOUT ---

st.title("✦ Consulting Slide Critique Tool")
st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["Critique Deck", "Knowledge Base"])

# --- TAB 1: CRITIQUE DECK ---
with tab1:
    col_header, col_clear = st.columns([3, 1])
    with col_header:
        st.header("Upload Presentation")
        st.write("Upload your PDF or PPTX deck and select the slide range to analyze")
    with col_clear:
        # Align button to bottom of header
        st.write("") 
        st.write("")
        if st.button("🗑️ Clear Workspace"):
            if clear_workspace():
                # Clear Session State relevant to uploads
                keys_to_clear = ["last_uploaded", "slide_paths", "current_pdf_path"]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.toast("Workspace cleared!", icon="✅")
                # We need to rerun to reset the file uploader state if possible, 
                # though st.rerun() is good.
                st.rerun()
    
    uploaded_pdf = st.file_uploader("Presentation File", type=["pdf", "pptx"], key="critique_upload")
    
    if uploaded_pdf:
        # Check if we've already processed this exact file
        if "last_uploaded" not in st.session_state or st.session_state["last_uploaded"] != uploaded_pdf.name:
            
            # It's a NEW file. Process it.
            temp_pdf_path = save_uploaded_file(uploaded_pdf)
            is_pptx = uploaded_pdf.name.lower().endswith(".pptx")
            
            if is_pptx:
                with st.spinner("Converting PPTX to PDF (this may take a moment)..."):
                    # Define a new path for the converted PDF
                    converted_pdf_path = os.path.splitext(temp_pdf_path)[0] + ".pdf"
                    
                    if Cloud_Converter.convert_pptx_to_pdf(temp_pdf_path, converted_pdf_path):
                        # Update temp_pdf_path to the new PDF
                        temp_pdf_path = converted_pdf_path
                    else:
                        st.error("Failed to convert PPTX to PDF. Check your API key or file.")
                        st.stop()
            
            with st.spinner("Processing PDF slides..."):
                # SAVE TO USER_IMG_DIR
                slide_paths = PDF_Translator.pdf_to_images(temp_pdf_path, output_folder=USER_IMG_DIR)
            
            # Update Session State
            st.session_state["last_uploaded"] = uploaded_pdf.name
            st.session_state["slide_paths"] = slide_paths
            st.session_state["current_pdf_path"] = temp_pdf_path
            
        else:
            # It's the SAME file. Load from cache.
            slide_paths = st.session_state["slide_paths"]
            temp_pdf_path = st.session_state.get("current_pdf_path", "")
            
        total_slides = len(slide_paths)
        st.success(f"Loaded {total_slides} slides.")

        # Range Selector
        col_range_1, col_range_2 = st.columns(2)
        with col_range_1:
            start_slide = st.number_input("From Slide", min_value=1, max_value=total_slides, value=1)
        with col_range_2:
            end_slide = st.number_input("To Slide", min_value=1, max_value=total_slides, value=min(total_slides, 10))

        # Search weighting Slider
        st.write("---")
        st.write("**Search Balance** (Influence on Reference Selection)")
        
        # Single slider with custom formatting
        # 0 = 100% Content (Text), 100 = 100% Layout (Visual)
        balance_val = st.slider(
            "Slide Balance",
            min_value=0,
            max_value=100,
            value=70,
            step=5,
            label_visibility="collapsed"
        )
        
        # Labels below slider
        col_lbl_left, col_lbl_right = st.columns([1,1])
        with col_lbl_left:
            st.text(f"Layout (Visual): {balance_val}%")
        with col_lbl_right:
            st.markdown(f"<div style='text-align: right'>Content (Text): {100-balance_val}%</div>", unsafe_allow_html=True)

        # Analysis Mode Toggle
        analysis_mode = st.radio(
            "Analysis Mode", 
            ["Full Critique (AI + Visuals)", "Visual Match Only (No AI Text)"], 
            horizontal=True
        )

        # Analyze Button
        if st.button("Analyze Slides", type="primary"):
            st.markdown("---")
            
            # Loop through selected range
            # Adjust for 0-based index vs 1-based user input
            for i in range(start_slide - 1, end_slide):
                current_slide_path = slide_paths[i]
                slide_filename = os.path.basename(current_slide_path)
                
                st.subheader(f"Analyzing Slide {i + 1}...")
                
                # Extract Text for this slide (Hybrid Search)
                query_text = ""
                try:
                    if temp_pdf_path and os.path.exists(temp_pdf_path):
                        # Use session state doc if possible or open new
                        # To be safe/simple inside loop:
                        with fitz.open(temp_pdf_path) as doc:
                            if i < len(doc):
                                query_text = doc[i].get_text()
                except Exception as e:
                    print(f"Text extract error: {e}")

                # Calculate visual weight (0.0 to 1.0)
                v_weight = balance_val / 100.0

                # Dynamic Layout based on mode
                if "Full" in analysis_mode:
                    # 3 Columns: User | Critique | References
                    col_result_left, col_result_mid, col_result_right = st.columns([1, 1.5, 1])
                else:
                    # 2 Columns: User | References
                    col_result_left, col_result_right = st.columns([1, 1])
                    col_result_mid = None # Flag to skip
                
                # 1. User Slide
                with col_result_left:
                    st.write("**YOUR SLIDE**")
                    st.image(current_slide_path, width="stretch", caption=f"Slide {i+1}")
                
                # 2. AI Critique (Only if Full Mode)
                if col_result_mid:
                    with col_result_mid:
                        st.write("**CRITIQUE & RECOMMENDATIONS**")
                        with st.spinner("Generating critique..."):
                            try:
                                critique_text = Slide_Critic.critique_slide(
                                    current_slide_path, 
                                    user_slide_text=query_text, 
                                    visual_weight=v_weight
                                )
                                st.markdown(critique_text)
                            except Exception as e:
                                st.error(f"Error during critique: {e}")
                
                # 3. Reference Examples
                with col_result_right:
                    st.write("**REFERENCE EXAMPLES**")
                    with st.spinner("Fetching similar slides..."):
                        try:
                            # query_similar_slides returns paths to similar images
                            # Logic updated to filter for GOLD_IMG_DIR inside VB_encoder or implicitly
                            
                            similar_slides = VB_encoder.query_similar_slides(
                                current_slide_path, 
                                input_text_content=query_text, 
                                n_results=3,
                                visual_weight=v_weight
                            )
                            
                            for idx, ref_path in enumerate(similar_slides):
                                # Sanitize path for Cloud/Local compatibility
                                if "slide_images" in ref_path:
                                    # Split to remove user-specific absolute path prefix
                                    # Keeps everything after "slide_images"
                                    rel_part = ref_path.split("slide_images")[-1].lstrip(os.sep)
                                    # Reconstruct using the current environment's BASE_DIR
                                    ref_path = os.path.join(BASE_DIR, "slide_images", rel_part)
                                
                                if os.path.exists(ref_path):
                                    st.image(ref_path, width="stretch", caption=f"Gold Standard #{idx+1}")
                                else:
                                    st.warning(f"Reference image missing: {os.path.basename(ref_path)}")
                                    
                        except Exception as e:
                            st.error(f"Error fetching references: {e}")
                
                st.markdown("---")


# --- TAB 2: KNOWLEDGE BASE ---
with tab2:
    st.header("Admin Access")
    passcode = st.text_input("Enter Passcode to unlock Knowledge Base:", type="password")
    
    if passcode == "MCT2025USC":
        st.success("Access Granted")
        st.divider()
        
        st.header("Add Gold Standard Decks")
        st.write("Upload high-quality presentations to train the critique system")
        
        uploaded_gold_pdfs = st.file_uploader("Gold Standard Presentation", type=["pdf", "pptx"], key="gold_upload", accept_multiple_files=True)
        
        if uploaded_gold_pdfs:
            st.success(f"Selected: **{len(uploaded_gold_pdfs)} files**")
            
        # Optional metadata input as per screenshot
        deck_tag = st.text_input("Deck Category/Tag (Optional)", placeholder="e.g., Market Analysis, Strategy, Financial")
        
        col_add, col_reset = st.columns([1, 1])
        
        with col_add:
            if st.button("Add to Memory", type="primary"):
                if uploaded_gold_pdfs:
                    
                    status_container = st.empty()
                    overall_progress_bar = st.progress(0)
                    
                    total_files = len(uploaded_gold_pdfs)
                    
                    for file_idx, uploaded_file in enumerate(uploaded_gold_pdfs):
                        status_container.info(f"Processing File {file_idx + 1}/{total_files}: {uploaded_file.name}")
                        
                        temp_gold_path = save_uploaded_file(uploaded_file)
                        
                        # Check for PPTX and convert if needed
                        if uploaded_file.name.lower().endswith(".pptx"):
                            status_container.info(f"Converting {uploaded_file.name} to PDF...")
                            converted_path = os.path.splitext(temp_gold_path)[0] + ".pdf"
                            if Cloud_Converter.convert_pptx_to_pdf(temp_gold_path, converted_path):
                                temp_gold_path = converted_path
                            else:
                                st.error(f"Failed to convert {uploaded_file.name}")
                                continue # Skip this file
                        
                        try:
                            # 1. Convert PDF to Images
                            gold_slide_paths = PDF_Translator.pdf_to_images(temp_gold_path, output_folder=GOLD_IMG_DIR)
                            
                            if not gold_slide_paths:
                                st.warning(f"No slides extracted from {uploaded_file.name}")
                            else:
                                status_container.info(f"Encoding {len(gold_slide_paths)} slides from {uploaded_file.name}...")
                                
                                # 2. Process slides one by one (Classic Mode)
                                count = 0
                                file_progress_bar = st.progress(0)
                                num_slides = len(gold_slide_paths)
                                
                                # Open PDF for text extraction
                                try:
                                    doc = fitz.open(temp_gold_path)
                                except:
                                    doc = None
                                
                                for s_idx, s_path in enumerate(gold_slide_paths):
                                    # Extract text
                                    slide_text = ""
                                    if doc and s_idx < len(doc):
                                        slide_text = doc[s_idx].get_text()
                                        
                                    if VB_encoder.add_slide_to_memory(s_path, text_content=slide_text):
                                        count += 1
                                    # Update inner progress bar
                                    file_progress_bar.progress((s_idx + 1) / num_slides)

                                st.toast(f"Indexed {count} slides!", icon="✅")
                                    
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {e}")
                        
                        # Update Overall Progress
                        overall_progress_bar.progress((file_idx + 1) / total_files)

                    status_container.success("All files processed successfully!")
                    status_container.success("All files processed successfully!")
                    st.toast("Batch upload successful!", icon="✅")

                else:
                    st.warning("Please upload at least one PDF file.")

        with col_reset:
            st.write("") # Spacer
            with st.expander("⚠️ Danger Zone", expanded=False):
                st.warning("This will delete all Gold Standard slides and wipe the AI memory.")
                
                confirm_text = st.text_input("Type 'CONFIRM RESET' to proceed:", key="reset_confirm")
                
                if confirm_text == "CONFIRM RESET":
                    if st.button("💣 WIPE EVERYTHING", type="primary"):
                        try:
                            # 1. Clear Database
                            if VB_encoder.reset_memory():
                                
                                # 2. Clear Files
                                if os.path.exists(GOLD_IMG_DIR):
                                    try:
                                        # Robust Clear: Delete contents instead of the folder itself
                                        for filename in os.listdir(GOLD_IMG_DIR):
                                            file_path = os.path.join(GOLD_IMG_DIR, filename)
                                            try:
                                                if os.path.isfile(file_path) or os.path.islink(file_path):
                                                    os.unlink(file_path)
                                                elif os.path.isdir(file_path):
                                                    shutil.rmtree(file_path)
                                            except Exception as inner_e:
                                                print(f"Skipped {filename}: {inner_e}")
                                    except Exception as e:
                                        st.warning(f"Warning during file cleanup: {e}")
                                
                                # Ensure directory exists
                                os.makedirs(GOLD_IMG_DIR, exist_ok=True)
                                    
                                st.toast("Knowledge Base wiped successfully!", icon="💥")
                                st.snow()
                                time.sleep(2) # Wait for animation
                                st.rerun()
                            else:
                                st.error("Failed to reset database.")
                        except Exception as e:
                            st.error(f"Error resetting: {e}")
                else:
                    st.button("Reset Knowledge Base", disabled=True, help="Type CONFIRM RESET above to enable")
    elif passcode:
        st.error("Incorrect Passcode. Access Denied.")
