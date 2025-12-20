import fitz
import os
import uuid

# Create a function to find pdf_path
def find_pdffolder_path():
    
    # Find the global path of this file
    global_path = os.path.dirname(os.path.abspath(__file__))
    
    # Find the pdf_folder_path
    pdf_folder_path = os.path.join(global_path, "Input_PDFs")
    
    return pdf_folder_path

def pdf_to_images(pdf_path, output_folder="slide_images"):
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")
    
    # Opening the PDF
    try: 
         doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return

    # Defining Zoom matrix (for 200% resolution)
    zoom_x = 2.0
    zoom_y = 2.0
    mat = fitz.Matrix(zoom_x, zoom_y)

    saved_images = []

    # Creating deck name variable
    deck_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Generate a short unique ID for this specific batch of slides
    batch_unique_id = str(uuid.uuid4())[:8]

    print(f"Processing '{deck_name}' (Batch ID: {batch_unique_id})...")
    
    # Looking through to convert each page to an image
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=mat)

        filename = f"{deck_name}_Slide_{page_num + 1}_{batch_unique_id}.png"
        file_path = os.path.join(output_folder, filename)

        pix.save(file_path)
        saved_images.append(file_path)

    return saved_images

# Testing Block
if __name__ == "__main__":

    # Finding the pdf_path
    pdf_path = os.path.join(find_pdffolder_path(), "kaufland_mctryouts.pdf")

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file {pdf_path} does not exist.")
        exit()  # <--- CHANGED THIS (stops the script)
    else:
        images = pdf_to_images(pdf_path, output_folder="slide_images/training_set"  )
        # 'images' is a list of file paths, so printing it might be huge. 
        # Let's just print the count:
        print(f"Success! Saved {len(images)} images.")