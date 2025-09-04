import os
import requests
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from dotenv import load_dotenv

load_dotenv()

# --- Configurations ---
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

app = Flask(__name__)
CORS(app)

# The health check is still useful
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route('/correct-text', methods=['POST'])
def correct_text_endpoint():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text_to_analyze = data['text']
    
    if not text_to_analyze.strip():
        return jsonify({"productName": None, "quantity": None, "description": None, "ingredients": None})

    prompt = (
        "You are an expert data extractor for consumer health products. "
        "Analyze the following OCR text. Your task is to extract: "
        "'productName', 'quantity', 'description', and 'ingredients'. "
        "Format your response as a JSON object. If a field is not found, its value must be null. "
        "Do not add any text outside of the single JSON object.\n\n"
        "--- OCR TEXT ---\n"
        f"{text_to_analyze}\n"
        "--- JSON OUTPUT ---"
    )
    
    try:
        response = model.generate_content(prompt)
        json_string = response.text.strip().replace('```json', '').replace('```', '').strip()
        structured_data = json.loads(json_string)
        return jsonify(structured_data)
    except Exception as e:
        print(f"AI Data Extraction Error: {e}")
        return jsonify({"error": f"AI failed to generate valid data. Details: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)