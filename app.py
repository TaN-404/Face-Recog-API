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
import face_recognition

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Assuming the model directory and device ID are fixed for the API
MODEL_DIR = "./models/anti_spoof_models"
DEVICE_ID = 0

def embeddings(name, path):
    image = cv2.imread(path)
    if image is None:
        return None, "Image not found or unable to load."
    embeddings = face_recognition.face_encodings(image)

    with open(os.path.join('db', f'{name}.pickle'), 'wb') as file:
        pickle.dump(embeddings, file)

    with open('./user_list.txt', 'a') as user_list_file:
        user_list_file.write(f'{name}\n')
    
def recognize(img, db_path):
    image = cv2.imread(img)
    if image is None:
        return None, "Image not found or unable to load."

    embeddings_unknown = face_recognition.face_encodings(image)
    if len(embeddings_unknown) == 0:
        return 'no_persons_found'
    else:
        embeddings_unknown = embeddings_unknown[0]

    db_dir = sorted(os.listdir(db_path))

    match = False
    j = 0
    while not match and j < len(db_dir):
        path_ = os.path.join(db_path, db_dir[j])

        file = open(path_, 'rb')
        embeddings = pickle.load(file)

        match = np.array(face_recognition.compare_faces([embeddings], embeddings_unknown)).any()
        j += 1

    if match:
        return db_dir[j - 1][:-7]
    else:
        return 'unknown_person'


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

@app.route('/register', methods = ['POST'])
def register():
    if 'name' not in request.form:
        return jsonify({'error': 'No name provided.'}), 400
    name = request.form['name']
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected.'}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('/tmp', filename)
        file.save(file_path)
        embeddings(name, file_path)
        return jsonify({'success': 'User registered successfully.'}), 200


@app.route('/login', methods=['POST'])
def login():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected.'}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('/tmp', filename)
        file.save(file_path)
        db_path = 'db'
        name = recognize(file_path, db_path )
        return jsonify({'success': f'{name} logged in successfully.'}), 200


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