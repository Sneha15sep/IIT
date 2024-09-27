import gradio as gr
import json
import torch
import numpy as np
import tempfile
import os
import re
from collections import Counter
from PIL import Image
import cv2
from transformers import TrOCRProcessor, AutoModelForVision2Seq


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
    processor = AutoProcessor.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForVision2Seq.from_pretrained(model_name, use_auth_token=hf_token)
    model.to(device)
    return processor, model

# Preprocess the image for better OCR performance
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)
    processed_image_path = image_path.replace('.png', '_processed.png')
    cv2.imwrite(processed_image_path, binary_image)
    return processed_image_path

# Perform OCR with General OCR and return plain text
def perform_ocr_general(image, use_gpu=False, hf_token=None):
    processor, model = initialize_general_ocr_model(use_gpu=use_gpu, hf_token=hf_token)
    
    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image.save(tmp_file.name)
        tmp_filepath = tmp_file.name

    # Preprocess the image before passing it to the model
    processed_image_path = preprocess_image(tmp_filepath)

    # Open the processed image
    image = Image.open(processed_image_path).convert("RGB")
    
    # Preprocess the image for the model
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(model.device)
    
    # Generate text from image pixels
    generated_ids = model.generate(pixel_values)
    extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if not extracted_text or len(extracted_text) < 3:
        return "No meaningful text extracted."
    
    return extracted_text

# Function to highlight keywords
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

# Gradio UI components
def ocr_and_search(image, hf_token, keywords, highlight_style, highlight_color):
    extracted_text = perform_ocr_general(image, hf_token=hf_token)
    
    if keywords:
        keywords = [kw.strip() for kw in keywords.split(",")]
        word_count, keyword_counts = analyze_text(extracted_text, keywords)
        
        highlighted_text = highlight_keywords(extracted_text, keywords, highlight_style, highlight_color)
        return extracted_text, word_count, keyword_counts, highlighted_text
    else:
        return extracted_text, 0, {}, extracted_text

# Building Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("# 📸 OCR & Document Search Web Application")
    gr.Markdown("### Extract and search text from images in Hindi and English")

    with gr.Row():
        image_input = gr.Image(label="Upload Image")
        hf_token = gr.Textbox(label="Hugging Face Token (for private models)", type="password")
    
    keywords_input = gr.Textbox(label="Enter keywords (comma-separated):")
    highlight_style = gr.Dropdown(choices=["Highlight with background color", "Underline", "Bold"], label="Highlight Style")
    highlight_color = gr.ColorPicker(label="Highlight Color", value="#FFFF00")
    
    with gr.Row():
        extract_button = gr.Button("Extract & Analyze")

    # Output components
    extracted_text_output = gr.Textbox(label="Extracted Text")
    word_count_output = gr.Number(label="Word Count")
    keyword_counts_output = gr.JSON(label="Keyword Counts")
    highlighted_text_output = gr.HTML(label="Highlighted Text")

    extract_button.click(
        ocr_and_search, 
        inputs=[image_input, hf_token, keywords_input, highlight_style, highlight_color], 
        outputs=[extracted_text_output, word_count_output, keyword_counts_output, highlighted_text_output]
    )

# Launching the Gradio app
app.launch()
