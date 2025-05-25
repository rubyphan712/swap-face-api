from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from fastapi.responses import StreamingResponse
import io

app = FastAPI()

@app.post("/swap")
async def swap_face(face_img: UploadFile, clothes_img: UploadFile):
    face_bytes = await face_img.read()
    clothes_bytes = await clothes_img.read()
    
    # Convert to images
    face_image = cv2.imdecode(np.frombuffer(face_bytes, np.uint8), cv2.IMREAD_COLOR)
    clothes_image = cv2.imdecode(np.frombuffer(clothes_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    from swap_logic import swap_faces  # example import if in another file
    swapped_image = swap_faces(face_image, clothes_image)

    
    _, img_encoded = cv2.imencode('.jpg', swapped_image)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")
