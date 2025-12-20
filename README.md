# Consulting Slide Critique Tool ✦

Try out the tool: https://mctslidecritic.streamlit.app/

An AI-powered application designed to critique presentation slides against "Gold Standard" examples. This tool uses **Google Gemini** for reasoning and a **Hybrid Search (Visual + Text)** system to find relevant reference slides from an internal knowledge base.

## Key Features

*   **Hybrid Search RAG**: Retrieves reference slides based on both **Visual Layout** (using OpenAI CLIP) and **Text Content** (using MiniLM).
*   **AI Critique**: Uses Google Gemini to generate specific, actionable feedback (Visual vs. Content) by comparing your slide to top-tier consulting examples.
*   **Gold Standard Memory**: "Knowledge Base" tab allowing admins to upload and index high-quality PDF/PPTX decks into the vector database.
*   **Privacy-Focused**: Runs locally or on Streamlit Cloud with secure API key management.
*   **PPTX Support**: Automatic conversion of PowerPoint files to PDF for analysis (requires CloudConvert API).


## Automated Slide Classifier (Computer Vision)

An optimized computer vision pipeline designed to classify slides into 7 structural archetypes (e.g., Data Charts, Process Flows).

**⚙️ Methodology**
*   **Architecture**: MobileNetV2 (Transfer Learning) with a custom Global Average Pooling head, selected for inference latency and parameter efficiency.
*   **Imbalance Handling**: Utilized training class weights (up to 5.0x for rare classes) and stratified validation to prevent bias towards text-heavy slides.

**🧪 Optimization & Inference**
*   **The Sweet Spot**: Using an automated checkpoint the model with the highest macro recall was selected. This means that that the average recall across all classes was the highest.
*   **Dynamic Per-Class Thresholding**: Abandoned global thresholds in favor of class-specific gates:
    *   *High Precision (0.8)* for **Strategic Text** to prevent hallucinations.
    *   *High Recall (0.3)* for **Charts/Flows** to capture visually diverse edge cases.

**📈 Custom Evaluation**
*   **Error Balance Sheets**: Developed custom visualization tools to distinguish between "Hallucinations" (False Positives) and "Missed Slides" (False Negatives), driving the dynamic threshold strategy.
*   **Cross-Contamination Matrix**: A heatmap specifically designed to show which classes were being confused with each other (e.g., verifying that Frameworks were often misclassified as Process Flows).

## Tech Stack


*   **Frontend**: Streamlit
*   **AI Model**: Google Gemini (via `google-generativeai`)
*   **Vector Database**: ChromaDB
*   **Embeddings**:
    *   Visual: `openai/clip-vit-base-patch32`
    *   Text: `sentence-transformers/all-MiniLM-L6-v2`
*   **PDF Processing**: PyMuPDF (`fitz`)
*   **Image Processing**: Pillow (PIL)

## Installation

### 1. System Dependencies
This tool requires **Poppler** for PDF conversions (used by `pdf2image` or similar utilities if configured, though `PyMuPDF` handles most rendering):

*   **Mac**: `brew install poppler`
*   **Linux (Debian/Ubuntu)**: `sudo apt-get install poppler-utils`

### 2. Python Dependencies
Install the required libraries from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. API Keys
You need the following API keys in a `secrets.json` file in the root directory (or in Streamlit Secrets for cloud deployment):

```json
{
    "GOOGLE_API_KEY": "your_gemini_api_key",
    "cloudconvert_api_key": "your_cloudconvert_api_key"
}
```

*   **Google Gemini**: For the critique generation.
*   **CloudConvert**: For converting `.pptx` uploads to PDF.

## Usage

Run the application locally:

```bash
streamlit run app.py
```

### How to Use
1.  **Critique Deck**: Upload your PDF or PPTX slide. Select a "Match Balance" (how much to weigh Visual Layout vs Text Content). Click **Analyze**.
2.  **Knowledge Base**: (Admin only) Enter the passcode to upload new "Gold Standard" decks into the memory.

## Deployment

Refer to `DEPLOYMENT_GUIDE.md` for instructions on deploying to Streamlit Community Cloud using **Git LFS** for the database.
