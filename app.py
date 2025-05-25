from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io
from simswap_api import swap_faces

app = FastAPI(title="Face Swap API using SimSwap")

@app.post("/swap")
async def swap_face(face_img: UploadFile = File(...), clothes_img: UploadFile = File(...)):
    # Read uploaded images
    face_bytes = await face_img.read()
    clothes_bytes = await clothes_img.read()
    
    # Convert to numpy arrays
    face_image = cv2.imdecode(np.frombuffer(face_bytes, np.uint8), cv2.IMREAD_COLOR)
    clothes_image = cv2.imdecode(np.frombuffer(clothes_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Perform face swap using SimSwap
    swapped_image = swap_faces(face_image, clothes_image)

    # Encode result as JPEG
    _, img_encoded = cv2.imencode('.jpg', swapped_image)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

@app.get("/")
def root():
    return {"message": "Face Swap API is running!"}
