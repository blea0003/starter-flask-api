from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from main import patient_registration_parser

app = Flask(__name__)
CORS(app)

@app.route('/patient_htr', methods=['POST'])
def patient_htr():
    """
    Request body
    - Image: *Handwritten filled document following template "docs/templates/Patient_Registration_v1.docx"* 

    Returns
    - Output of patient_registration_parser() on received image
    """
    # Check if request contains an 'image' field
    if 'image' not in request.files:
        return "No image file provided", 400

    # Check if file has allowed extension
    file = request.files['image']
    allowed_extensions = ['jpg', 'jpeg', 'png']
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return "Invalid file format. Supported formats: jpg, jpeg, png", 400
    
    try:
        # Read image data and decode it
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Parse document
        output = patient_registration_parser(img, model="keras_ocr", debug=False)

        return jsonify(output)
    
    except Exception as e:
        return f'Error: {str(e)}', 500


@app.route('/patient_htr_sample', methods=['POST'])
def patient_htr_sample():
    """
    A sample version of the patient_htr function.
    Returns the same valid output example regardless of input image.
    """
    # Check if request contains an 'image' field
    if 'image' not in request.files:
        return 'No image file provided', 400

    temp_obj = {
        "ic": "31081960101984",
        "firstName": "Jerry",
        "lastName": "Seinfeld",
        "dob": "1960-08-31",
        "gender": "Male",
        "nationality": "American",
        "phoneNo": " 5558383",
        "email": "jerryseinfeld@gmail.com",
        "emergencyNo": "601170735766",
        "emergencyRemarks": "You know George",
        "remarks": "What's the deal with arthritis?"
    }

    return jsonify(temp_obj)



if __name__ == "__main__":
    app.run(debug=True)