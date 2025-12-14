import streamlit as st
import os
import pandas as pd

# Configuration Map
DATASETS = {
    "Training Set": {
        "folder": "slide_images/training_set",
        "csv": "slide_images/training_set_tags.csv"
    },
    "Validation Set": {
        "folder": "slide_images/validation_set",
        "csv": "slide_images/validation_set_tags.csv"
    }
}

st.set_page_config(layout="wide", page_title="Gold Standard Tagger")

# --- Custom CSS to shrink buttons ---
st.markdown("""
<style>
    /* Target buttons in the sidebar to make them smaller */
    [data-testid="stSidebar"] button {
        font-size: 10px !important;
        padding: 2px 5px !important;
        min-height: 0px !important;
        height: auto !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
st.sidebar.title("Dataset")
selected_dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    list(DATASETS.keys()),
    index=0
)

# Detect Dataset Change and Reset State
if 'last_dataset_name' not in st.session_state:
    st.session_state.last_dataset_name = selected_dataset_name

if st.session_state.last_dataset_name != selected_dataset_name:
    st.session_state.current_image_index = 0
    st.session_state.tags_input_val = ""
    st.session_state.last_dataset_name = selected_dataset_name
    st.rerun()

# Set globals for current selection
IMAGE_FOLDER = DATASETS[selected_dataset_name]["folder"]
CSV_FILE = DATASETS[selected_dataset_name]["csv"]

st.title(f"Tagger: {selected_dataset_name}")

# --- Helper Functions ---

def get_all_images():
    """Returns a sorted list of image filenames in the folder."""
    if not os.path.exists(IMAGE_FOLDER):
        st.error(f"Folder not found: {IMAGE_FOLDER}")
        return []
    
    files = [f for f in os.listdir(IMAGE_FOLDER) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    return sorted(files)

def load_existing_tags():
    """Loads the CSV into a dict {filename: tags_string}."""
    if not os.path.exists(CSV_FILE):
        return {}
    
    try:
        df = pd.read_csv(CSV_FILE)
        # Ensure columns exist
        if 'filename' not in df.columns or 'tags' not in df.columns:
            return {}
        
        # Convert to dict
        return pd.Series(df.tags.values, index=df.filename).to_dict()
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return {}

def save_tag(filename, tags):
    """Appends or updates a tag in the CSV."""
    # Read existing to ensure we don't have duplicates or to update
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame(columns=['filename', 'tags'])

    # Update or Append
    if filename in df['filename'].values:
        df.loc[df['filename'] == filename, 'tags'] = tags
    else:
        new_row = pd.DataFrame({'filename': [filename], 'tags': [tags]})
        df = pd.concat([df, new_row], ignore_index=True)

    # Ensure dir exists for CSV if not
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    df.to_csv(CSV_FILE, index=False)

# --- Main Logic ---

# 1. Load Data
all_images = get_all_images()
existing_tags_dict = load_existing_tags()

# 2. Initialize Session State
if 'current_image_index' not in st.session_state:
    # Find first untagged image index
    start_index = 0
    for i, img in enumerate(all_images):
        if img not in existing_tags_dict:
            start_index = i
            break
    st.session_state.current_image_index = start_index

# Safety check if index out of bounds (can happen during switch before reset fully propagates if race condition, but rerun handles it)
if st.session_state.current_image_index >= len(all_images) and len(all_images) > 0:
    st.session_state.current_image_index = 0

if 'tags_input_val' not in st.session_state:
    # Initialize tags for the first image if it exists
    if st.session_state.current_image_index < len(all_images):
        first_img = all_images[st.session_state.current_image_index]
        existing = existing_tags_dict.get(first_img, "")
        st.session_state.tags_input_val = str(existing) if pd.notna(existing) else ""
    else:
        st.session_state.tags_input_val = ""

# --- Callbacks ---

def load_tags_for_current_index():
    """Updates the input buffer with tags for the current image index."""
    idx = st.session_state.current_image_index
    if 0 <= idx < len(all_images):
        filename = all_images[idx]
        # Reloading dict to ensure we have latest save
        current_tags_dict = load_existing_tags() 
        existing = current_tags_dict.get(filename, "")
        st.session_state.tags_input_val = str(existing) if pd.notna(existing) else ""

def prev_image():
    """Move to previous image and load its tags."""
    if st.session_state.current_image_index > 0:
        st.session_state.current_image_index -= 1
        load_tags_for_current_index()

def next_image():
    """Move to next image and load its tags."""
    if st.session_state.current_image_index < len(all_images) - 1:
        st.session_state.current_image_index += 1
        load_tags_for_current_index()

def jump_to_untagged():
    """Finds the first image without tags and jumps there."""
    # Reload tags to be sure
    current_tags = load_existing_tags()
    for i, img in enumerate(all_images):
        if img not in current_tags:
            st.session_state.current_image_index = i
            load_tags_for_current_index()
            return
    st.warning("No untagged images found!")

def add_tag(tag):
    """Callback to append a tag to the current input."""
    current = st.session_state.tags_input_val.strip()
    if current:
        # Avoid duplicates
        cur_list = [t.strip() for t in current.split(',')]
        if tag not in cur_list:
            if current.endswith(','):
                 st.session_state.tags_input_val = current + " " + tag
            else:
                 st.session_state.tags_input_val = current + ", " + tag
    else:
        st.session_state.tags_input_val = tag

def save_and_next():
    """Callback to save tags and move to next."""
    idx = st.session_state.current_image_index
    if 0 <= idx < len(all_images):
        filename = all_images[idx]
        tags_to_save = st.session_state.tags_input_val
        
        if tags_to_save:
            save_tag(filename, tags_to_save)
            st.success(f"Saved tags for {filename}")
            
            # Move to next if possible
            if idx < len(all_images) - 1:
                st.session_state.current_image_index += 1
                load_tags_for_current_index()
            else:
                st.balloons()
                st.success("You have reached the end of the images!")
        else:
            st.warning("Please enter at least one tag.")

# --- Sidebar: Show Existing Tags ---
with st.sidebar:
    st.divider()
    st.header("Existing Tags")
    
    # ALWAYS load from Training Set for the sidebar buttons
    training_csv = DATASETS["Training Set"]["csv"]
    visible_tags = set()
    
    if os.path.exists(training_csv):
        try:
            df_train = pd.read_csv(training_csv)
            if 'tags' in df_train.columns:
                 for tags_str in df_train['tags'].dropna():
                    if isinstance(tags_str, str):
                        tags = [t.strip() for t in tags_str.split(',') if t.strip()]
                        visible_tags.update(tags)
        except Exception as e:
            st.error(f"Error loading training tags: {e}")

    if visible_tags:
        sorted_tags = sorted(list(visible_tags))
        st.write("Click to add:")
        
        # Render buttons in a grid
        cols_per_row = 4
        # Create rows
        for i in range(0, len(sorted_tags), cols_per_row):
            cols = st.columns(cols_per_row)
            # Get the batch for this row
            batch = sorted_tags[i:i+cols_per_row]
            
            for idx, tag in enumerate(batch):
                cols[idx].button(tag, on_click=add_tag, args=(tag,), width="stretch")

    else:
        st.write("No tags found in Training Set.")

# --- Main UI ---

# Progress
current_idx = st.session_state.current_image_index
total = len(all_images)

# If we are somehow out of bounds (e.g. empty folder), handle gracefully
if total == 0:
    st.warning("No images found in the folder.")
else:
    # Ensure index is within bounds
    if current_idx >= total:
        current_idx = total - 1
        st.session_state.current_image_index = current_idx

    # Calculate untagged count specifically
    untagged_count = len([img for img in all_images if img not in existing_tags_dict])
    st.progress((total - untagged_count) / total, text=f"Progress: {total - untagged_count}/{total} tagged. Viewing image {current_idx + 1} of {total}.")

    current_image_file = all_images[current_idx]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        image_path = os.path.join(IMAGE_FOLDER, current_image_file)
        st.image(image_path, width="stretch")
        st.caption(f"Filename: {current_image_file}")

    with col2:
        st.header("Tagging")
        
        st.text_input(
            "Enter tags (comma separated)", 
            key="tags_input_val"
        )
        
        # Navigation Buttons
        b_col1, b_col2, b_col3 = st.columns([1, 1, 1])
        
        with b_col1:
            st.button("Total Back", on_click=prev_image, disabled=(current_idx == 0), width="stretch")
            
        with b_col2:
            st.button("Jump to Untagged", on_click=jump_to_untagged, width="stretch")
        
        with b_col3:
            st.button("Save & Next", on_click=save_and_next, width="stretch")

    st.divider()

    # Optional overview at bottom
    if st.checkbox("Show all tags table"):
        st.dataframe(pd.read_csv(CSV_FILE))
