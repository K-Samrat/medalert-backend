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
    if not text_to_analyze.strip():
        return None
    
    prompt = (
        "You are an expert data extractor for health products. Analyze the following text extracted from a product's packaging. "
        "CRITICAL INSTRUCTION: You MUST NOT invent, guess, infer, or add any information that is not explicitly written in the provided text. Your entire response must be grounded in the source text.\n\n"
        "Your task is to identify and extract the following information: the product's name, a brief description (especially noting consumption instructions), and a list of all ingredients. "
        "Format your response as a JSON object only. The JSON object should have three keys: 'productName', 'description', and 'ingredients'. "
        "If a piece of information is not found in the text, its value should be null. Do not add any commentary or introductory text outside of the JSON object.\n\n"
        "Here is the text:\n---\n"
        f"{text_to_analyze}\n"
        "---\n\n"
        "JSON Output:"
    )
    
    try:
        response = model.generate_content(prompt)
        json_string = response.text.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(json_string)
    except Exception as e:
        print(f"AI Data Extraction Error: {e}")
        return {"error": "Failed to parse AI response."}

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
        return jsonify({'error': 'No selected files'}), 400
    
    all_raw_text = ""
    for file in files:
        try:
            image_bytes = file.read()
            raw_text = get_ocr_text(image_bytes)
            all_raw_text += raw_text + "\n\n"
        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")

    structured_data = extract_structured_data(all_raw_text)

    if structured_data:
        return jsonify(structured_data)
    else:
        return jsonify({'error': "Could not extract any data from the image(s)."}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)