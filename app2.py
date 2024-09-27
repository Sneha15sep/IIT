import json
import torch
import streamlit as st
import numpy as np
import tempfile
import os
import re
from collections import Counter
from PIL import Image
import cv2  # Import OpenCV for image processing
from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # Update imports

# Utility function to convert numpy types to native Python types
def convert_to_native_type(data):
    if isinstance(data, (np.ndarray, list)):
        return [convert_to_native_type(item) for item in data]
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    return data

# Check if GPU is available
def is_gpu_available():
    return torch.cuda.is_available()

# Initialize the General OCR model
def initialize_general_ocr_model(model_name='microsoft/trocr-base-printed', use_gpu=False, hf_token=None):
    device = torch.device("cuda" if use_gpu and is_gpu_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained(model_name, use_auth_token=hf_token)  # Use TrOCRProcessor
    model = VisionEncoderDecoderModel.from_pretrained(model_name, use_auth_token=hf_token)  # Use VisionEncoderDecoderModel
    model.to(device)
    return processor, model

# Preprocess the image for better OCR performance
def preprocess_image(image_path):
    # Load the image with OpenCV
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding
    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)
    # Save the processed image to a temporary location
    processed_image_path = image_path.replace('.png', '_processed.png')
    cv2.imwrite(processed_image_path, binary_image)
    return processed_image_path

# Perform OCR with General OCR and return plain text
def perform_ocr_general(image_path, use_gpu=False, hf_token=None):
    processor, model = initialize_general_ocr_model(use_gpu=use_gpu, hf_token=hf_token)
    
    # Preprocess the image before passing it to the model
    processed_image_path = preprocess_image(image_path)
    
    # Open the processed image
    image = Image.open(processed_image_path).convert("RGB")
    
    # Preprocess the image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(model.device)
    
    # Generate text from image pixels
    generated_ids = model.generate(pixel_values)
    extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Check if the extracted text is meaningful
    if not extracted_text or len(extracted_text) < 3:  # Assuming text should be longer than 3 characters
        return "No meaningful text extracted."
    
    return extracted_text

# Function to highlight keywords with chosen style
def highlight_keywords(text, keywords, highlight_style, highlight_color):
    if highlight_style == "Highlight with background color":
        pattern = re.compile(f"({'|'.join(map(re.escape, keywords))})", re.IGNORECASE)
        highlighted_text = pattern.sub(f'<span style="background-color: {highlight_color};">\\1</span>', text)
    elif highlight_style == "Underline":
        pattern = re.compile(f"({'|'.join(map(re.escape, keywords))})", re.IGNORECASE)
        highlighted_text = pattern.sub(r'<u>\1</u>', text)
    elif highlight_style == "Bold":
        pattern = re.compile(f"({'|'.join(map(re.escape, keywords))})", re.IGNORECASE)
        highlighted_text = pattern.sub(r'<b>\1</b>', text)
    else:
        highlighted_text = text  # No highlighting

    return highlighted_text

# Function to analyze text
def analyze_text(text, keywords):
    words = text.split()
    word_count = len(words)
    keyword_counts = Counter(word.lower() for word in words if word.lower() in [kw.lower() for kw in keywords])
    return word_count, keyword_counts

def main():
    # Set page configuration
    st.set_page_config(page_title="📸 OCR & Document Search", layout="wide", page_icon="📄")

    # Custom CSS for stylish, formal design
    st.markdown(""" 
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

        .stApp {
            font-family: 'Roboto', sans-serif;
        }
        .title {
            color: #2c3e50;
            text-align: center;
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 0.5em;
            letter-spacing: 0.05em;
        }
        .subtitle {
            text-align: center;
            font-size: 1.5em;
            margin-bottom: 2em;
            font-style: italic;
        }
        .stButton > button {
            background-color: #00b894;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 1em;
            transition: background-color 0.3s;
        }
        .stButton > button:hover {
            background-color: #00a47a;
        }
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #ccc;
        }
        .card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .section {
            background-color: #2d3436;
            color: #dfe6e9;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .footer {
            background-color: #34495e;
            color: white;
            text-align: center;
            padding: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header Image
    st.image("Optical Character Recognition.png", use_column_width=True)

    # Title and Description with formal design
    st.markdown('<p class="title">📸 OCR & Document Search Web Application</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Extract and search text from images in Hindi and English</p>', unsafe_allow_html=True)

    # Sidebar for Upload and Settings
    st.sidebar.header("🔧 Settings")

    # Hugging Face Token input
    hf_token = st.sidebar.text_input("🔑 Hugging Face Token (for private models)", type="password")

    # Color Picker for Highlighting
    highlight_color = st.sidebar.color_picker("Select Highlight Color", "#FFFF00")

    # Keyword Highlight Style Option
    st.sidebar.markdown("### 🔍 Keyword Highlighting")
    highlight_style = st.sidebar.selectbox(
        "Choose Highlight Style",
        ("Highlight with background color", "Underline", "Bold")
    )

    # Download Options
    st.sidebar.markdown("### 📥 Download Options")
    download_format = st.sidebar.selectbox(
        "Select Download Format",
        ("Plain Text", "JSON")
    )

    # File Uploader (Allowing only a single file upload)
    uploaded_file = st.sidebar.file_uploader("📂 Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_filepath = tmp_file.name

        # Display the uploaded image
        st.image(uploaded_file, caption='🖼️ Uploaded Image', use_column_width=True, clamp=True)

        # Perform OCR with a progress bar
        with st.spinner('🔄 Performing OCR...'):
            extracted_text = perform_ocr_general(tmp_filepath, hf_token=hf_token)

        # Display Extracted Text in a card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Extracted Text:")
        st.write(extracted_text)

        # Download Options
        if download_format == "Plain Text":
            st.download_button(
                label="Download as Plain Text",
                data=extracted_text,
                file_name='extracted_text.txt',
                mime='text/plain'
            )

        # Keyword Analysis and Highlighting
        keywords_input = st.text_input("🔑 Enter keywords (comma-separated):")
        if keywords_input:
            keywords = [kw.strip() for kw in keywords_input.split(",")]
            word_count, keyword_counts = analyze_text(extracted_text, keywords)

            # Highlighting logic
            highlighted_text = highlight_keywords(extracted_text, keywords, highlight_style, highlight_color)

            # Displaying keyword analysis results
            st.markdown("### Keyword Analysis:")
            st.write(f"Total Words: {word_count}")
            st.write("Keyword Counts:")
            for keyword, count in keyword_counts.items():
                st.write(f"{keyword}: {count}")

            # Display highlighted text
            st.markdown("### Highlighted Text:")
            st.markdown(highlighted_text, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Footer Section
    st.markdown('<div class="footer">📄 Created with ❤️ for OCR & Document Search</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
