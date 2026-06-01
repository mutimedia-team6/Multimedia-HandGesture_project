import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union


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

_MODEL_PATH = Path(__file__).resolve().parent / "model" / "fusion_mobilenetv3_landmark_aug14kdetect_best.pth"

_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------------------------------------------------------------
# Heuristic decision layer (inlined to avoid extra dependency)
# ---------------------------------------------------------------------

CLASS_NA = 0
CLASS_FIST = 1
CLASS_LIKE = 2
CLASS_OK = 3
CLASS_ONE = 4
CLASS_PALM = 5

CLASS_NAMES: Dict[int, str] = {
    CLASS_NA: "N/A",
    CLASS_FIST: "fist",
    CLASS_LIKE: "like",
    CLASS_OK: "ok",
    CLASS_ONE: "one",
    CLASS_PALM: "palm",
}

VALID_CLASSES = (CLASS_FIST, CLASS_LIKE, CLASS_OK, CLASS_ONE, CLASS_PALM)

WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

FINGER_JOINTS = {
    "thumb": (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

LONG_FINGERS = ("index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class HeuristicConfig:
    temperature: float = 1.5
    max_entropy: float = 1.35
    min_confidence: Dict[int, float] = None  # type: ignore[assignment]
    min_margin: Dict[int, float] = None  # type: ignore[assignment]
    use_landmark_rules: bool = True
    reject_when_landmark_invalid: bool = True
    ok_thumb_index_close: float = 0.65
    palm_min_spread: float = 0.45
    finger_extension_extra: float = 0.05

    def __post_init__(self):
        if self.min_confidence is None:
            object.__setattr__(
                self,
                "min_confidence",
                {
                    CLASS_FIST: 0.45,
                    CLASS_LIKE: 0.45,
                    CLASS_OK: 0.45,
                    CLASS_ONE: 0.45,
                    CLASS_PALM: 0.45,
                },
            )
        if self.min_margin is None:
            object.__setattr__(
                self,
                "min_margin",
                {
                    CLASS_FIST: 0.08,
                    CLASS_LIKE: 0.08,
                    CLASS_OK: 0.08,
                    CLASS_ONE: 0.08,
                    CLASS_PALM: 0.08,
                },
            )


DEFAULT_CONFIG = HeuristicConfig()


def _as_numpy_1d(x: Union[np.ndarray, list, tuple]) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)


def softmax(logits: Union[np.ndarray, list, tuple], temperature: float = 1.0) -> np.ndarray:
    z = _as_numpy_1d(logits)
    temperature = max(float(temperature), 1e-6)
    z = z / temperature
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


def looks_like_probability_vector(x: np.ndarray) -> bool:
    if x.ndim != 1:
        return False
    if np.any(~np.isfinite(x)):
        return False
    if np.any(x < -1e-6) or np.any(x > 1.0 + 1e-6):
        return False
    return abs(float(np.sum(x)) - 1.0) < 1e-3


def to_six_class_probabilities(
    model_output: Union[np.ndarray, list, tuple],
    config: HeuristicConfig = DEFAULT_CONFIG,
    assume_logits: Optional[bool] = None,
) -> np.ndarray:
    raw = _as_numpy_1d(model_output)

    if raw.size not in (5, 6):
        raise ValueError(
            f"Expected model output size 5 or 6, got shape {np.asarray(model_output).shape}."
        )

    if assume_logits is None:
        is_prob = looks_like_probability_vector(raw)
    else:
        is_prob = not assume_logits

    if is_prob:
        probs = raw.astype(np.float32)
        probs = probs / max(float(np.sum(probs)), 1e-8)
    else:
        probs = softmax(raw, temperature=config.temperature)

    if probs.size == 5:
        probs6 = np.zeros(6, dtype=np.float32)
        probs6[1:] = probs
        probs = probs6

    probs = probs.astype(np.float32)
    probs = probs / max(float(np.sum(probs)), 1e-8)
    return probs


def entropy(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    return float(-np.sum(p * np.log(p + 1e-8)))


def top_two(probs: np.ndarray) -> Tuple[int, float, int, float, float]:
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    order = np.argsort(p)
    top_class = int(order[-1])
    second_class = int(order[-2])
    top_prob = float(p[top_class])
    second_prob = float(p[second_class])
    margin = top_prob - second_prob
    return top_class, top_prob, second_class, second_prob, margin


def valid_landmarks(landmarks: Optional[np.ndarray]) -> bool:
    if landmarks is None:
        return False
    lm = np.asarray(landmarks, dtype=np.float32)
    if lm.ndim != 2:
        return False
    if lm.shape[0] != 21:
        return False
    if lm.shape[1] < 2:
        return False
    if np.any(~np.isfinite(lm[:, :2])):
        return False
    return True


def xy_landmarks(landmarks: np.ndarray) -> np.ndarray:
    return np.asarray(landmarks, dtype=np.float32)[:, :2]


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    bc = np.asarray(c, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    denom = max(float(np.linalg.norm(ba) * np.linalg.norm(bc)), 1e-8)
    cos_value = float(np.dot(ba, bc) / denom)
    cos_value = float(np.clip(cos_value, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_value)))


def palm_scale(lm: np.ndarray) -> float:
    wrist_to_middle = dist(lm[WRIST], lm[MIDDLE_MCP])
    index_to_pinky = dist(lm[INDEX_MCP], lm[PINKY_MCP])
    return max(wrist_to_middle, index_to_pinky, 1e-6)


def palm_center(lm: np.ndarray) -> np.ndarray:
    return np.mean(
        lm[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]], axis=0
    )


def finger_extended(lm: np.ndarray, finger: str, config: HeuristicConfig = DEFAULT_CONFIG) -> bool:
    s = palm_scale(lm)

    if finger == "thumb":
        _cmc, _mcp, ip, tip = FINGER_JOINTS["thumb"]
        center = palm_center(lm)

        tip_from_center = dist(lm[tip], center)
        ip_from_center = dist(lm[ip], center)
        tip_from_wrist = dist(lm[tip], lm[WRIST])
        ip_from_wrist = dist(lm[ip], lm[WRIST])

        return (
            tip_from_center > ip_from_center + config.finger_extension_extra * s
            and tip_from_wrist > ip_from_wrist + 0.02 * s
        )

    if finger not in LONG_FINGERS:
        raise ValueError(f"Unknown finger: {finger}")

    mcp, pip, dip, tip = FINGER_JOINTS[finger]

    pip_angle = angle_degrees(lm[mcp], lm[pip], lm[dip])
    dip_angle = angle_degrees(lm[pip], lm[dip], lm[tip])

    tip_from_wrist = dist(lm[tip], lm[WRIST])
    pip_from_wrist = dist(lm[pip], lm[WRIST])

    straight_by_angle = pip_angle > 145.0 and dip_angle > 140.0
    extended_by_distance = tip_from_wrist > pip_from_wrist + config.finger_extension_extra * s

    return bool(straight_by_angle and extended_by_distance)


def finger_states(lm: np.ndarray, config: HeuristicConfig = DEFAULT_CONFIG) -> Dict[str, bool]:
    return {
        "thumb": finger_extended(lm, "thumb", config),
        "index": finger_extended(lm, "index", config),
        "middle": finger_extended(lm, "middle", config),
        "ring": finger_extended(lm, "ring", config),
        "pinky": finger_extended(lm, "pinky", config),
    }


def long_finger_extension_count(states: Dict[str, bool]) -> int:
    return int(sum(bool(states[f]) for f in LONG_FINGERS))


def thumb_index_distance_ratio(lm: np.ndarray) -> float:
    return dist(lm[THUMB_TIP], lm[INDEX_TIP]) / palm_scale(lm)


def finger_spread_ratio(lm: np.ndarray) -> float:
    return dist(lm[INDEX_TIP], lm[PINKY_TIP]) / palm_scale(lm)


def landmark_rule_pass(
    pred_class: int,
    landmarks: Optional[np.ndarray],
    config: HeuristicConfig = DEFAULT_CONFIG,
) -> bool:
    if pred_class == CLASS_NA:
        return True

    if not valid_landmarks(landmarks):
        return not config.reject_when_landmark_invalid

    lm = xy_landmarks(landmarks)
    states = finger_states(lm, config)

    thumb = states["thumb"]
    index = states["index"]
    middle = states["middle"]
    ring = states["ring"]
    pinky = states["pinky"]

    long_count = long_finger_extension_count(states)

    if pred_class == CLASS_FIST:
        return long_count <= 1

    if pred_class == CLASS_LIKE:
        return thumb and long_count <= 1

    if pred_class == CLASS_OK:
        close_ratio = thumb_index_distance_ratio(lm)
        return close_ratio < config.ok_thumb_index_close

    if pred_class == CLASS_ONE:
        return index and not middle and not ring and not pinky

    if pred_class == CLASS_PALM:
        spread = finger_spread_ratio(lm)
        return index and middle and ring and pinky and spread > config.palm_min_spread

    return False


def final_decision(
    model_output: Union[np.ndarray, list, tuple],
    landmarks: Optional[np.ndarray],
    config: HeuristicConfig = DEFAULT_CONFIG,
    assume_logits: Optional[bool] = None,
    return_debug: bool = False,
):
    probs = to_six_class_probabilities(model_output, config, assume_logits=assume_logits)

    top_class, top_prob, second_class, second_prob, margin = top_two(probs)
    ent = entropy(probs)

    debug = {
        "probs": probs,
        "top_class": top_class,
        "top_name": CLASS_NAMES.get(top_class, "unknown"),
        "top_prob": top_prob,
        "second_class": second_class,
        "second_name": CLASS_NAMES.get(second_class, "unknown"),
        "second_prob": second_prob,
        "margin": margin,
        "entropy": ent,
        "reason": "accepted",
    }

    if top_class == CLASS_NA:
        debug["reason"] = "model_top_is_NA"
        return (CLASS_NA, debug) if return_debug else CLASS_NA

    if top_class not in VALID_CLASSES:
        debug["reason"] = "invalid_top_class"
        return (CLASS_NA, debug) if return_debug else CLASS_NA

    min_conf = float(config.min_confidence.get(top_class, 0.75))
    if top_prob < min_conf:
        debug["reason"] = f"low_confidence({top_prob:.3f} < {min_conf:.3f})"
        return (CLASS_NA, debug) if return_debug else CLASS_NA

    min_margin = float(config.min_margin.get(top_class, 0.15))
    if margin < min_margin:
        debug["reason"] = f"low_margin({margin:.3f} < {min_margin:.3f})"
        return (CLASS_NA, debug) if return_debug else CLASS_NA

    if ent > config.max_entropy:
        debug["reason"] = f"high_entropy({ent:.3f} > {config.max_entropy:.3f})"
        return (CLASS_NA, debug) if return_debug else CLASS_NA

    if config.use_landmark_rules:
        if not landmark_rule_pass(top_class, landmarks, config):
            debug["reason"] = "landmark_rule_failed"
            return (CLASS_NA, debug) if return_debug else CLASS_NA

    return (int(top_class), debug) if return_debug else int(top_class)


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
    lm = np.asarray(landmarks, dtype=np.float32)

    if lm.ndim != 2 or lm.shape[0] != 21 or lm.shape[1] < 2:
        lm_xy = np.zeros((21, 2), dtype=np.float32)
    else:
        lm_xy = lm[:, :2]

        wrist = lm_xy[0, :]
        lm_xy = lm_xy - wrist
        max_dist = np.max(np.abs(lm_xy))
        if max_dist > 0:
            lm_xy = lm_xy / max_dist

    tensor = torch.tensor(lm_xy, dtype=torch.float32).unsqueeze(0)
    return tensor.to(_DEVICE)

def _model_forward(cropped_img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    model = _load_model_once()

    img_tensor = _preprocess_image(cropped_img)
    lm_tensor = _preprocess_landmarks(landmarks)

    with torch.no_grad():
        logits = model(img_tensor, lm_tensor)

    return logits.squeeze(0).cpu().numpy()

# -----------------------------------------------------------------------------
# Optional scoring utility for validation tuning
# -----------------------------------------------------------------------------

def assignment_raw_score(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Compute a raw score similar to the assignment rule.

    Based on the spec:
    - Correctly predicting a 5-class target: +1
    - False trigger or misclassification: -2

    We treat valid gesture -> N/A as 0 here because the spec emphasizes false
    trigger penalty and does not clearly say that missed valid gestures are -2.
    If your TA clarifies otherwise, update this function.
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    score = 0
    for t, p in zip(y_true, y_pred):
        t = int(t)
        p = int(p)

        if t in VALID_CLASSES:
            if p == t:
                score += 1
            elif p == CLASS_NA:
                score += 0
            else:
                score -= 2
        elif t == CLASS_NA:
            if p == CLASS_NA:
                score += 0
            else:
                score -= 2
        else:
            # Unknown labels should not appear, but treat them as N/A-like.
            if p != CLASS_NA:
                score -= 2

    return int(score)
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
