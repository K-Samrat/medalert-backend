import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image # <--- THIS LINE WAS MISSING
import io

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

@app.route('/ocr', methods=['POST'])
def ocr():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Prepare the request for the OCR.space API
            api_url = 'https://api.ocr.space/parse/image'
            api_key = 'K87359756488957' # Use your actual API key here

            payload = {'apikey': api_key}
            files = {'file': (file.filename, file.read(), file.content_type)}
            
            # Send the request
            response = requests.post(api_url, files=files, data=payload)
            response.raise_for_status()

            result = response.json()

            if result.get('IsErroredOnProcessing'):
                return jsonify({'error': result.get('ErrorMessage', ['OCR processing error'])[0]}), 500
            
            parsed_text = result.get('ParsedResults', [{}])[0].get('ParsedText', '')
            
            return jsonify({'text': parsed_text})
        except requests.exceptions.RequestException as e:
            return jsonify({'error': f"API request failed: {e}"}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)