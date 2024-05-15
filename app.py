from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import argparse
import warnings
import time
from werkzeug.utils import secure_filename
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Assuming the model directory and device ID are fixed for the API
MODEL_DIR = "./models/anti_spoof_models"
DEVICE_ID = 0

def test(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, "Image not found or unable to load."
    model_test = AntiSpoofPredict(DEVICE_ID)
    image_cropper = CropImage()
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    for model_name in os.listdir(MODEL_DIR):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        model_prediction = model_test.predict(img, os.path.join(MODEL_DIR, model_name))
        prediction += model_prediction

    label = np.argmax(prediction)
    return label

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected.'}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('/tmp', filename)
        file.save(file_path)
        label = test(file_path)
        if label is None:
            return jsonify({'error': 'Failed to process image.'}), 500
        return jsonify({'result': bool(label == 1)})

if __name__ == '__main__':
    app.run(debug=True)