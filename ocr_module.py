import easyocr
import json
import torch

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
            'bounding_box': res[0],
            'text': res[1],
            'confidence': res[2]
        })
    
    # Convert to pretty JSON format
    return json.dumps(structured_output, ensure_ascii=False, indent=2)

# Example usage:
if __name__ == "__main__":
    image_path = "path_to_image.jpg"  # Replace with your image path
    use_gpu = False  # Set this to True if you want to use GPU (if available)
    
    # Perform OCR and get text
    extracted_text = perform_ocr_easyocr(image_path, use_gpu=use_gpu)
    print("Extracted Text:\n", extracted_text)
    
    # Get extracted text as JSON
    json_output = extract_text_json(image_path, use_gpu=use_gpu)
    print("Extracted Text (JSON):\n", json_output)
