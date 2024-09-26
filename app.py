import easyocr
import json
import torch
import streamlit as st
import numpy as np
import tempfile
import os
import re
from collections import Counter

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

# Initialize the EasyOCR reader dynamically based on language and GPU availability
def initialize_reader(languages=None, use_gpu=False):
    if languages is None:
        languages = ['en', 'hi']  # Default to English and Hindi
    reader = easyocr.Reader(languages, gpu=use_gpu)
    return reader

# Perform OCR with EasyOCR and return plain text
def perform_ocr_easyocr(image_path, use_gpu=False, languages=['en', 'hi']):
    reader = initialize_reader(languages, use_gpu=use_gpu)
    results = reader.readtext(image_path, detail=0, paragraph=True)
    extracted_text = ' '.join(results)
    return extracted_text

# Extract text and return as plain text (multi-line format)
def extract_text(image_path, use_gpu=False, languages=['en', 'hi']):
    reader = initialize_reader(languages, use_gpu=use_gpu)
    results = reader.readtext(image_path, detail=0, paragraph=True)
    extracted_text = '\n'.join(results)
    return extracted_text

# Extract text with bounding boxes and confidence, return as JSON
def extract_text_json(image_path, use_gpu=False, languages=['en', 'hi']):
    reader = initialize_reader(languages, use_gpu=use_gpu)
    results = reader.readtext(image_path, detail=1)

    # Structuring the output
    structured_output = []
    for res in results:
        structured_output.append({
            'bounding_box': convert_to_native_type(res[0]),  # Ensure it's converted to a native type
            'text': res[1],
            'confidence': convert_to_native_type(res[2])  # Ensure it's converted to a native type
        })

    # Convert to pretty JSON format
    return json.dumps(structured_output, ensure_ascii=False, indent=2)

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
            extracted_text = extract_text(tmp_filepath)
            json_output = extract_text_json(tmp_filepath)  # Get JSON output

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
        elif download_format == "JSON":
            st.download_button(
                label="Download as JSON",
                data=json_output,
                file_name='extracted_text.json',
                mime='application/json'
            )

        # Keyword Analysis and Highlighting
        keywords_input = st.text_input("🔑 Enter keywords (comma-separated):")
        if keywords_input:
            keywords = [kw.strip() for kw in keywords_input.split(",")]
            word_count, keyword_counts = analyze_text(extracted_text, keywords)

            # Highlighting logic
            highlighted_text = highlight_keywords(extracted_text, keywords, highlight_style, highlight_color)

            # Displaying results
            st.markdown(f"### Results for keywords: {', '.join(keywords)}")
            st.write(f"Total Word Count: {word_count}")
            for keyword, count in keyword_counts.items():
                st.write(f"Keyword: **{keyword}** - Count: {count}")

            st.markdown("### Highlighted Text:")
            st.markdown(highlighted_text, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Clean up temporary file
        os.remove(tmp_filepath)

    # Footer
    st.markdown('<div class="footer"> Made with ❤️ by Sneha Agrawal</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
