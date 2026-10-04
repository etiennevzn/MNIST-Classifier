import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model import MNISTClassifier

MODEL_PATH = ROOT_DIR / "mnist_model.pth"
IMAGE_SIZE = 28


def center_by_mass(image):
    total = image.sum()
    if total == 0:
        return image

    rows, cols = np.indices(image.shape)
    center_row = (rows * image).sum() / total
    center_col = (cols * image).sum() / total

    target = (IMAGE_SIZE - 1) / 2
    shift_row = int(round(target - center_row))
    shift_col = int(round(target - center_col))

    padded = np.pad(image, IMAGE_SIZE)
    top = IMAGE_SIZE - shift_row
    left = IMAGE_SIZE - shift_col
    return padded[top:top + IMAGE_SIZE, left:left + IMAGE_SIZE]


def preprocess(pixels):
    image = np.array(pixels, dtype=np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE) / 255.0
    image = center_by_mass(image)
    tensor = torch.from_numpy(np.ascontiguousarray(image))
    return tensor.unsqueeze(0).unsqueeze(0)


class Predictor:
    def __init__(self, model_path=MODEL_PATH, neurons=128, hidden_layers=2):
        self.model = MNISTClassifier(neurons, hidden_layers)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def predict(self, pixels):
        tensor = preprocess(pixels)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)

        return probabilities.squeeze(0).numpy()
