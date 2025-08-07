import os
import requests
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# --- AI Configuration ---
# Get API keys from environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OCR_SPACE_API_KEY = os.getenv('OCR_SPACE_API_KEY')

# Configure the Gemini client
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')
# ----------------------

app = Flask(__name__)
CORS(app)

def correct_text_with_ai(text_to_correct):
    """Uses the Gemini AI to correct OCR text."""
    if not text_to_correct.strip():
        return ""
    
    prompt = (
        "The following text was extracted from an image using OCR and may contain spelling and formatting errors. "
        "Please correct the text to make it clean, readable, and accurate. Preserve the original line breaks if possible. "
        "Do not add any commentary or introductory phrases, just provide the corrected text.\n\n"
        "Messy Text:\n---\n"
        f"{text_to_correct}\n"
        "---\n\n"
        "Corrected Text:"
    )
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"AI Correction Error: {e}")
        return f"AI Correction Failed. Raw Text: {text_to_correct}"


@app.route('/ocr', methods=['POST'])
def ocr():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # --- Step 1: Raw OCR Extraction ---
            ocr_api_url = 'https://api.ocr.space/parse/image'
            
            payload = {'apikey': OCR_SPACE_API_KEY, 'OCREngine': '2'}
            files = {'file': (file.filename, file.read(), file.content_type)}
            
            response = requests.post(ocr_api_url, files=files, data=payload)
            response.raise_for_status()
            ocr_result = response.json()

            if ocr_result.get('IsErroredOnProcessing'):
                return jsonify({'error': ocr_result.get('ErrorMessage', ['OCR processing error'])[0]}), 500
            
            raw_text = ocr_result.get('ParsedResults', [{}])[0].get('ParsedText', '')
            
            # --- Step 2: AI Correction ---
            clean_text = correct_text_with_ai(raw_text)

            return jsonify({'text': clean_text})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)