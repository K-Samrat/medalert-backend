import os
import requests
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import json
from dotenv import load_dotenv

load_dotenv()

# --- Configurations ---
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OCR_SPACE_API_KEY = os.getenv('OCR_SPACE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

app = Flask(__name__)
CORS(app)

def extract_structured_data(text_to_analyze):
    if not text_to_analyze.strip():
        return None
    
    # --- THIS IS OUR NEW, MORE POWERFUL PROMPT ---
    prompt = (
        "You are an expert data extractor specializing in pharmaceutical and medical products. "
        "Analyze the following text extracted from a product's packaging. "
        "Your task is to identify and extract the following information with high accuracy: "
        "1. 'productName': The main brand name of the product (e.g., 'Flexon'). "
        "2. 'description': A brief summary of the product's use, warnings, or dosage instructions (e.g., 'Keep out of reach of children.'). "
        "3. 'ingredients': A list of all active ingredients with their specific quantities (e.g., 'Ibuprofen 400mg', 'Paracetamol 325mg'). "
        "Format your response as a single, clean JSON object. The JSON object must have three keys: 'productName' (string), 'description' (string), and 'ingredients' (an array of strings). "
        "If a piece of information is not found in the text, its value should be null. Do not add any commentary or introductory text outside of the JSON object.\n\n"
        "Here is the text:\n---\n"
        f"{text_to_analyze}\n"
        "---\n\n"
        "JSON Output:"
    )
    
    try:
        response = model.generate_content(prompt)
        # Clean up the response to ensure it is valid JSON
        json_string = response.text.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(json_string)
    except Exception as e:
        print(f"AI Data Extraction Error: {e}")
        return {"error": "Failed to parse the AI's response."}

def get_ocr_text(image_bytes, engine_number=2):
    """Function to call the OCR.space API."""
    ocr_api_url = 'https://api.ocr.space/parse/image'
    payload = {
        'apikey': OCR_SPACE_API_KEY,
        'OCREngine': str(engine_number)
    }
    files = {'file': ('image.jpg', image_bytes)}
    response = requests.post(ocr_api_url, files=files, data=payload)
    response.raise_for_status()
    result = response.json()

    if result.get('IsErroredOnProcessing'):
        print(f"Engine {engine_number} Error: {result.get('ErrorMessage')}")
        return ""
    return result.get('ParsedResults', [{}])[0].get('ParsedText', '')

@app.route('/ocr', methods=['POST'])
def ocr():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    all_raw_text = ""
    files = request.files.getlist('files[]') # Get all files from the request

    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected files'}), 400

    for file in files:
        try:
            image_bytes = file.read()
            raw_text = get_ocr_text(image_bytes)
            all_raw_text += raw_text + "\n\n" # Combine text from all images
        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")

    # Send the combined raw text to the AI for structured data extraction
    structured_data = extract_structured_data(all_raw_text)

    if structured_data:
        return jsonify(structured_data)
    else:
        return jsonify({'error': "Could not extract any structured data from the image(s)."}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)