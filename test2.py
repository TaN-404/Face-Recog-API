import os
import cv2
import numpy as np
import argparse
import warnings
import time
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


def test(image_path, model_dir, device_id):
    # Load the image from the path
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Create an instance of the AntiSpoofPredict class
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    # Get bounding box for the image
    image_bbox = model_test.get_bbox(image)
    if image_bbox is None:
        raise ValueError("No bounding box found in the image.")

    # Crop the image based on the bounding box
    x, y, w, h = image_bbox
    cropped_image = image[y:y+h, x:x+w]

    # Initialize prediction and test speed
    prediction = np.zeros((1, 3))
    test_speed = 0

    # Sum the prediction from each model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": cropped_image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time() - start

    # Calculate average prediction speed
    average_speed = test_speed / len(os.listdir(model_dir))

    # Determine the result of the prediction
    label = np.argmax(prediction)
    confidence = prediction[0][label] / 2  # Example confidence calculation

    # Print or return the results
    print(f"Predicted label: {label}, Confidence: {confidence:.2f}, Average processing time: {average_speed:.2f} seconds")

    return label, confidence, average_speed

if __name__ == "__main__":
    desc = "Test script for face recognition"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--model_dir", type=str, default="./models/anti_spoof_models", help="Directory containing models")
    parser.add_argument("--image_path", type=str, default="./samples/img3.jpeg", help="Image to test")
    args = parser.parse_args()
    test(args.image_path, args.model_dir, args.device_id)