import pytesseract
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

# If you did not add Tesseract to your system's PATH during installation,
# you must tell Python where the tesseract.exe file is.
# Uncomment the line below if you have issues later.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize the Flask app
app = Flask(__name__)
CORS(app) # Allow communication from your React app

@app.route('/ocr', methods=['POST'])
def ocr():
    # Check if a file was posted in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    # Check if the user selected a file
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        try:
            # Read the image file
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))

            # Use Pytesseract to extract text from the image
            text = pytesseract.image_to_string(image, lang='eng')
            
            # Return the extracted text in a JSON format
            return jsonify({'text': text})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the backend server on port 5000
    app.run(debug=True, port=5000)