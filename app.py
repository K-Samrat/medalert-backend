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

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

def extract_structured_data(text_to_analyze):
    if not text_to_analyze or not text_to_analyze.strip():
        return {"error": "No text was extracted from the image to analyze."}
    
    prompt = (
        "You are an expert data extractor for consumer products. Analyze the following OCR text. "
        "Your task is to extract: 'productName', 'quantity', 'description', and 'ingredients'. "
        "Format your response as a JSON object. If a field is not found, its value must be null. "
        "Do not add any text outside of the single JSON object.\n\n"
        "--- OCR TEXT ---\n"
        f"{text_to_analyze}\n"
        "--- JSON OUTPUT ---"
    )
    
    try:
        response = model.generate_content(prompt)
        json_string = response.text.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(json_string)
    except Exception as e:
        print(f"AI Data Extraction Error: {e}")
        return {"error": f"AI failed to generate valid data. Details: {str(e)}"}

def get_ocr_text(image_bytes, engine_number=2):
    ocr_api_url = 'https://api.ocr.space/parse/image'
    payload = {'apikey': OCR_SPACE_API_KEY, 'OCREngine': str(engine_number)}
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
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    all_raw_text = ""
    for file in files:
        try:
            # Added a specific try-except for image processing
            try:
                image_bytes = file.read()
                # Verify image can be opened by Pillow to prevent deep errors
                Image.open(io.BytesIO(image_bytes)).verify()
            except Exception as image_error:
                print(f"Error processing image file {file.filename}: {image_error}")
                continue # Skip this file and move to the next

            raw_text = get_ocr_text(image_bytes)
            all_raw_text += raw_text + "\n\n"
        except Exception as e:
            print(f"Error in OCR or network for file {file.filename}: {e}")

    structured_data = extract_structured_data(all_raw_text)

    if structured_data:
        return jsonify(structured_data)
    else:
        return jsonify({'error': "Could not extract any data from the image(s)."}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)