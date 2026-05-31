import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from heuristic import final_decision


# ---------------------------------------------------------------------
# Model definition: must match training notebook
# ---------------------------------------------------------------------

class MobileNetV3SmallFeatureExtractor(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()

        # IMPORTANT:
        # Use weights=None during inference.
        # The checkpoint already contains the trained weights, so we should not
        # download ImageNet weights in the official Colab runtime.
        backbone = mobilenet_v3_small(weights=None)

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        self.projector = nn.Sequential(
            nn.Linear(576, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.projector(x)
        return x


class LandmarkMLP(nn.Module):
    def __init__(self, input_dim=42, hidden_dim=64, output_dim=128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),

            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, landmarks):
        if landmarks.dim() == 3:
            landmarks = landmarks.view(landmarks.size(0), -1)

        return self.mlp(landmarks)


class GestureFusionModel(nn.Module):
    def __init__(self, image_dim=128, landmark_dim=128, num_classes=6):
        super().__init__()

        self.image_encoder = MobileNetV3SmallFeatureExtractor(output_dim=image_dim)
        self.landmark_encoder = LandmarkMLP(output_dim=landmark_dim)

        fusion_dim = image_dim + landmark_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, img, landmarks):
        image_feature = self.image_encoder(img)
        landmark_feature = self.landmark_encoder(landmarks)
        fusion_feature = torch.cat([image_feature, landmark_feature], dim=1)
        logits = self.classifier(fusion_feature)
        return logits


# ---------------------------------------------------------------------
# Global model loading
# ---------------------------------------------------------------------

_MODEL = None
_DEVICE = torch.device("cpu")

_MODEL_PATH = Path(__file__).resolve().parent / "model" / "fusion_mobilenetv3_landmark_best.pth"

_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _load_model_once():
    global _MODEL

    if _MODEL is None:
        model = GestureFusionModel(
            image_dim=128,
            landmark_dim=128,
            num_classes=6,
        )

        state_dict = torch.load(_MODEL_PATH, map_location=_DEVICE)
        model.load_state_dict(state_dict)
        model.to(_DEVICE)
        model.eval()

        _MODEL = model

    return _MODEL


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

def _preprocess_image(cropped_img: np.ndarray) -> torch.Tensor:
    """
    cropped_img: RGB hand crop, np.ndarray, usually H x W x 3.
    return: tensor shape [1, 3, 224, 224]
    """
    img = np.asarray(cropped_img)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.shape[-1] == 4:
        img = img[..., :3]

    if img.dtype != np.uint8:
        # Support float images in [0,1] or [0,255].
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img).convert("RGB")
    tensor = _IMAGE_TRANSFORM(pil_img).unsqueeze(0)
    return tensor.to(_DEVICE)


def _preprocess_landmarks(landmarks: np.ndarray) -> torch.Tensor:
    """
    Match training code:
    landmarks are loaded as raw np.float32 values and directly fed to the model.
    No wrist-relative normalization is applied here.
    """
    lm = np.asarray(landmarks, dtype=np.float32)

    if lm.ndim != 2 or lm.shape[0] != 21 or lm.shape[1] < 2:
        lm_xy = np.zeros((21, 2), dtype=np.float32)
    else:
        lm_xy = lm[:, :2]

    tensor = torch.tensor(lm_xy, dtype=torch.float32).unsqueeze(0)
    return tensor.to(_DEVICE)


def _model_forward(cropped_img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    model = _load_model_once()

    img_tensor = _preprocess_image(cropped_img)
    lm_tensor = _preprocess_landmarks(landmarks)

    with torch.no_grad():
        logits = model(img_tensor, lm_tensor)

    return logits.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------
# Required interface
# ---------------------------------------------------------------------

def predict(cropped_img: np.ndarray, landmarks: np.ndarray) -> int:
    """
    Required by assignment spec.

    Return:
        0 = N/A
        1 = fist
        2 = like
        3 = ok
        4 = one
        5 = palm
    """
    model_output = _model_forward(cropped_img, landmarks)

    pred = final_decision(
        model_output=model_output,
        landmarks=landmarks,
        assume_logits=True,
    )

    return int(pred)