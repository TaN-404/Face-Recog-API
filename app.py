from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
import os
import cv2
import numpy as np
import warnings
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
import pickle
from deepface import DeepFace
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings('ignore')

MODEL_DIR = os.getenv("MODEL_DIR", "./models/anti_spoof_models")
DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))
REGISTERED_FACES_DIR = os.getenv("REGISTERED_FACES_DIR", "registered_faces")
ENCODINGS_DIR = os.getenv("ENCODINGS_DIR", "db_deep")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))

os.makedirs(REGISTERED_FACES_DIR, exist_ok=True)
os.makedirs(ENCODINGS_DIR, exist_ok=True)

# Global in-memory cache for encodings
memory_encodings = {}

def load_all_encodings():
    """Loads encodings from disk to memory at startup."""
    global memory_encodings
    encodings = {}
    if os.path.exists(ENCODINGS_DIR):
        for file_name in os.listdir(ENCODINGS_DIR):
            if file_name.endswith('.pkl'):
                user_name = file_name.split('.')[0]
                with open(os.path.join(ENCODINGS_DIR, file_name), 'rb') as f:
                    encodings[user_name] = pickle.load(f)
    memory_encodings = encodings
    print(f"Loaded {len(memory_encodings)} encodings.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load encodings
    await run_in_threadpool(load_all_encodings)
    yield
    # Shutdown

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def save_face(image: np.ndarray, user_name: str):
    face_path = os.path.join(REGISTERED_FACES_DIR, f"{user_name}.jpg")
    cv2.imwrite(face_path, image)

def save_encoding(user_name: str, encoding):
    encoding_path = os.path.join(ENCODINGS_DIR, f"{user_name}.pkl")
    with open(encoding_path, 'wb') as f:
        pickle.dump(encoding, f)
    # Update memory cache
    memory_encodings[user_name] = encoding

def process_anti_spoof_check(image: np.ndarray) -> bool:
    model_test = AntiSpoofPredict(DEVICE_ID)
    image_cropper = CropImage()
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    
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
    return bool(label == 1)

def verify_face_against_db(input_encoding) -> str:
    """Verifies an input encoding against the cached dictionary of encodings."""
    for user_name, registered_encoding in memory_encodings.items():
        try:
            result = DeepFace.verify(input_encoding, registered_encoding, model_name='VGG-Face', distance_metric='cosine')
            if result.get("verified", False):
                return user_name
        except Exception as e:
            print(f"Error verifying against user {user_name}: {e}")
            pass
    return None

def process_registration(image: np.ndarray, user_name: str):
    save_face(image, user_name)
    encoding_result = DeepFace.represent(image, model_name='VGG-Face')
    encoding = encoding_result[0]["embedding"]
    save_encoding(user_name, encoding)

def process_login(image: np.ndarray) -> str:
    # 1. Anti Spoof
    is_real = process_anti_spoof_check(image)
    if not is_real:
        raise HTTPException(status_code=401, detail="Login failed: Spoof detected")
    
    # 2. Extract encoding
    encoding_result = DeepFace.represent(image, model_name='VGG-Face')
    input_encoding = encoding_result[0]["embedding"]
    
    # 3. Verify
    user_name = verify_face_against_db(input_encoding)
    if user_name is None:
        raise HTTPException(status_code=401, detail="Login failed: User not recognized")
    
    return user_name

@app.post("/register")
async def register(name: str = Form(...), image: UploadFile = File(...)):
    image_bytes = await image.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img_cv2 = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    if img_cv2 is None:
        raise HTTPException(status_code=400, detail="Invalid image input")

    await run_in_threadpool(process_registration, img_cv2, name)
    return {"message": "User registered successfully"}

@app.post("/login")
async def login(image: UploadFile = File(...)):
    image_bytes = await image.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img_cv2 = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img_cv2 is None:
        raise HTTPException(status_code=400, detail="Invalid image input")

    user_name = await run_in_threadpool(process_login, img_cv2)
    return {"message": f"Login successful for user {user_name}"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
