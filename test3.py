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

def check_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image.")
        return False
    height, width, channel = image.shape
    if width / height != 3 / 4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True

def test(model_dir, device_id, image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
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
        start = time.time()
        model_prediction = model_test.predict(img, os.path.join(model_dir, model_name))
        prediction += model_prediction
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
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./models/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--image_path",
        type=str,
        default="./samples/img3.jpeg",
        help="image used to test")
    args = parser.parse_args()
    test(args.model_dir, args.device_id, args.image_path)