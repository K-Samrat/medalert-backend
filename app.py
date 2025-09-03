import os
import requests
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
from dotenv import load_dotenv # <--- THIS LINE WAS MISSING

# Load variables from the .env file
load_dotenv()

# --- Configurations ---
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OCR_SPACE_API_KEY = os.getenv('OCR_SPACE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

### --- Flask App Setup ---
app = Flask(__name__)
CORS(app)

def correct_text_with_ai(text_to_correct):
    if not text_to_correct.strip():
        return ""
    
    prompt = (
        "The following text was extracted from an image using OCR and contains errors. "
        "Correct all spelling and grammatical mistakes to make it clean and accurate. "
        "Do not add any commentary, just provide the corrected text.\n\n"
        "Messy Text:\n---\n"
        f"{text_to_correct}\n"
        "---\n\n"
        "Corrected Text:"
    )
    
    try:
        response = model.generate_content(prompt)
        if response.parts:
            return response.text
        else:
            return f"AI model returned no content. Raw Text: {text_to_correct}"
    except Exception as e:
        print(f"AI Correction Error: {e}")
        return f"AI Correction Failed. Raw Text: {text_to_correct}"

def get_ocr_text(image_bytes, engine_number):
    ocr_api_url = 'https://api.ocr.space/parse/image'
    payload = {
        'apikey': OCR_SPACE_API_KEY,
        'OCREngine': str(engine_number),
        'scale': 'true',
        'detectOrientation': 'true'
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
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            image_bytes = file.read()
            raw_text = get_ocr_text(image_bytes, 2)
            
            if not raw_text.strip():
                print("Engine 2 failed, trying Engine 5...")
                raw_text = get_ocr_text(image_bytes, 5)

            clean_text = correct_text_with_ai(raw_text)

            return jsonify({'text': clean_text or "Could not extract any text from the image."})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)