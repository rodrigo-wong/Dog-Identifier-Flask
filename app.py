from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import cv2
import os
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
CORS(app)


# Load the trained model
modelFile = "./model/dogs.keras"
model = tf.keras.models.load_model(modelFile)

# Define input shape for the model
inputShape = (331, 331)

# Load the labels (categories)
allLabels = np.load("./temp/allLabels.npy")
categories = np.unique(allLabels)

# Function to prepare the image for the model
def prepareImage(img):
    resized = cv2.resize(img, inputShape, interpolation=cv2.INTER_AREA)
    imgResult = np.expand_dims(resized, axis=0)
    imgResult = imgResult / 255.0  # Normalize the image
    return imgResult

@app.route('/predict', methods=['POST'])
def predict_breed():
    if 'image' not in request.files:
        return jsonify({"error": "No image file found in the request."}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    # Secure and save the file temporarily
    filename = secure_filename(image_file.filename)
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, filename)
    image_file.save(temp_path)

    try:
        # Read and prepare the image
        img = cv2.imread(temp_path)
        if img is None:
            return jsonify({"error": "Unable to read the image."}), 400

        imageForModel = prepareImage(img)

        # Predict the class of the image
        resultArray = model.predict(imageForModel)
        answers = np.argmax(resultArray, axis=1)

        # Get the corresponding label and confidence
        predicted_breed = categories[answers[0]]
        confidence = resultArray[0][answers[0]]

        # Return the result as JSON
        return jsonify({
            "breed": str(predicted_breed),
            "confidence": f"{confidence:.2%}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        os.rmdir(temp_dir)

if __name__ == '__main__':
    app.run(debug=True)
