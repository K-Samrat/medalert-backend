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
    
    # --- FINAL, MOST ROBUST PROMPT ---
    prompt = (
        "You are a highly advanced data extraction AI for consumer goods. Analyze the OCR text from a product image. "
        "Your goal is to extract key information and structure it as a JSON object. "
        "The keys must be: 'productName', 'quantity', 'description', 'ingredients', and 'nutritionFacts'.\n"
        "1.  **productName**: Find the primary brand or product name (e.g., 'Flexon', 'Cheerios').\n"
        "2.  **quantity**: Find the net quantity of the product. Look for terms like '15 tablets', 'Net Wt', '500ml', '75g'.\n"
        "3.  **description**: Extract warnings ('Keep out of reach of children'), dosage, or short usage instructions.\n"
        "4.  **ingredients**: Extract the list of chemical ingredients for medicines OR the list of food ingredients.\n"
        "5.  **nutritionFacts**: Extract all data from the 'Nutrition Facts' panel if it exists.\n"
        "CRITICAL: Always return a valid JSON object. If any field is not found, its value MUST be null. Do not add any text or markdown formatting outside of the single JSON object."
        "\n\n--- OCR TEXT TO ANALYZE ---\n"
        f"{text_to_analyze}"
        "\n\n--- JSON OUTPUT ---"
    )
    
    try:
        response = model.generate_content(prompt)
        # Clean up the response to ensure it is valid JSON
        json_string = response.text.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(json_string)
    except Exception as e:
        print(f"AI Data Extraction Error: {e}")
        return {"error": f"AI failed to generate valid data. Details: {str(e)}"}

# ... (The rest of the app.py file is unchanged and correct) ...
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