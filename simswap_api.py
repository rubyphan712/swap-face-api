import sys
from pathlib import Path

# Add SimSwap to Python path
sys.path.append(str(Path(__file__).resolve().parent / "SimSwap"))

import torch
from PIL import Image
import cv2
import numpy as np
from models.models import create_model
from util.face_align import align_face
from options.test_options import TestOptions

opt = TestOptions().parse()  # Use default config
opt.name = 'people'
opt.Arc_path = './arcface_model/arcface_checkpoint.tar'
opt.pic_a_path = ''
opt.pic_b_path = ''
opt.output_path = './output.jpg'
opt.isTrain = False
opt.use_mask = False
opt.no_simswaplogo = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(opt)
model.eval()

def swap_faces(face_image: np.ndarray, clothes_image: np.ndarray) -> np.ndarray:
    cv2.imwrite('face.jpg', face_image)
    cv2.imwrite('clothes.jpg', clothes_image)

    aligned_face_pil = align_face('face.jpg')

    with torch.no_grad():
        model.forward(aligned_face_pil, 'clothes.jpg')

    swapped_image = cv2.imread('output.jpg')
    return swapped_image
