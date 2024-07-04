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
import pickle
from deepface import DeepFace

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Assuming the model directory and device ID are fixed for the API
MODEL_DIR = "./models/anti_spoof_models"
DEVICE_ID = 0

# Directory to store registered faces and encodings
REGISTERED_FACES_DIR = "registered_faces"
ENCODINGS_DIR = "db_deep"

# Ensure the directories exist
if not os.path.exists(REGISTERED_FACES_DIR):
    os.makedirs(REGISTERED_FACES_DIR)

if not os.path.exists(ENCODINGS_DIR):
    os.makedirs(ENCODINGS_DIR)

def save_face(image, user_name):
    face_path = os.path.join(REGISTERED_FACES_DIR, f"{user_name}.jpg")
    cv2.imwrite(face_path, image)

def save_encoding(user_name, encoding):
    encoding_path = os.path.join(ENCODINGS_DIR, f"{user_name}.pkl")
    with open(encoding_path, 'wb') as f:
        pickle.dump(encoding, f)

def load_encodings():
    encodings = {}
    for file_name in os.listdir(ENCODINGS_DIR):
        if file_name.endswith('.pkl'):
            user_name = file_name.split('.')[0]
            with open(os.path.join(ENCODINGS_DIR, file_name), 'rb') as f:
                encodings[user_name] = pickle.load(f)
    return encodings


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

@app.route('/register', methods=['POST'])
def register():
    user_name = request.form['name']
    image_file = request.files['image']
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    save_face(image, user_name)
    
    encoding = DeepFace.represent(image, model_name='VGG-Face')[0]["embedding"]
    save_encoding(user_name, encoding)
    
    return jsonify({"message": "User registered successfully"}), 200

@app.route('/login', methods=['POST'])
def login():
    image_file = request.files['image']
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    encodings = load_encodings()
    input_encoding = DeepFace.represent(image, model_name='VGG-Face')[0]["embedding"]
    
    for user_name, registered_encoding in encodings.items():
        result = DeepFace.verify(input_encoding, registered_encoding, model_name='VGG-Face', distance_metric='cosine')
        if result["verified"]:
            return jsonify({"message": f"Login successful for user {user_name}"}), 200
    
    return jsonify({"message": "Login failed"}), 401


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