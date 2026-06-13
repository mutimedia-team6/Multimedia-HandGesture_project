from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union


# ---------------------------------------------------------------------
# Model definition: must match training notebook
# ---------------------------------------------------------------------

class LandmarkOnlyModel(nn.Module):
    def __init__(self, input_dim=42, num_classes=6):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, num_classes),
        )

    def forward(self, landmarks):
        if landmarks.dim() == 3:
            landmarks = landmarks.view(landmarks.size(0), -1)
        return self.mlp(landmarks)


# ---------------------------------------------------------------------
# Global model loading
# ---------------------------------------------------------------------

_MODEL = None
_DEVICE = torch.device("cpu")

_MODEL_PATH = Path(__file__).resolve().parent / "model" / "landmark_only.pth"


# ---------------------------------------------------------------------
# Heuristic decision layer
# ---------------------------------------------------------------------

CLASS_NA   = 0
CLASS_FIST = 1
CLASS_LIKE = 2
CLASS_OK   = 3
CLASS_ONE  = 4
CLASS_PALM = 5

CLASS_NAMES: Dict[int, str] = {
    CLASS_NA:   "N/A",
    CLASS_FIST: "fist",
    CLASS_LIKE: "like",
    CLASS_OK:   "ok",
    CLASS_ONE:  "one",
    CLASS_PALM: "palm",
}

VALID_CLASSES = (CLASS_FIST, CLASS_LIKE, CLASS_OK, CLASS_ONE, CLASS_PALM)

WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP  = 3
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
    "thumb":  (THUMB_CMC,  THUMB_MCP,  THUMB_IP,  THUMB_TIP),
    "index":  (INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring":   (RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP),
    "pinky":  (PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP),
}

LONG_FINGERS = ("index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class HeuristicConfig:
    temperature: float = 1.5
    max_entropy: float = 1.35
    min_confidence: Dict[int, float] = None   # type: ignore[assignment]
    min_margin: Dict[int, float] = None       # type: ignore[assignment]
    use_landmark_rules: bool = True
    reject_when_landmark_invalid: bool = True
    ok_thumb_index_close: float = 0.65
    palm_min_spread: float = 0.65
    finger_extension_extra: float = 0.05

    def __post_init__(self):
        if self.min_confidence is None:
            object.__setattr__(self, "min_confidence", {
                CLASS_FIST: 0.45,
                CLASS_LIKE: 0.45,
                CLASS_OK:   0.45,
                CLASS_ONE:  0.45,
                CLASS_PALM: 0.45,
            })
        if self.min_margin is None:
            object.__setattr__(self, "min_margin", {
                CLASS_FIST: 0.08,
                CLASS_LIKE: 0.08,
                CLASS_OK:   0.08,
                CLASS_ONE:  0.08,
                CLASS_PALM: 0.08,
            })


DEFAULT_CONFIG = HeuristicConfig(
    temperature=1.5,
    max_entropy=1.35,
    min_confidence={
        CLASS_FIST: 0.45,
        CLASS_LIKE: 0.45,
        CLASS_OK:   0.45,
        CLASS_ONE:  0.42,
        CLASS_PALM: 0.65,
    },
    min_margin={
        CLASS_FIST: 0.05,
        CLASS_LIKE: 0.05,
        CLASS_OK:   0.05,
        CLASS_ONE:  0.03,
        CLASS_PALM: 0.12,
    },
    ok_thumb_index_close=0.65,
    palm_min_spread=0.65,
    finger_extension_extra=0.05,
)


def _as_numpy_1d(x: Union[np.ndarray, list, tuple]) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)


def softmax(logits: Union[np.ndarray, list, tuple], temperature: float = 1.0) -> np.ndarray:
    z = _as_numpy_1d(logits)
    temperature = max(float(temperature), 1e-6)
    z = z / temperature - np.max(z / temperature)
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
        raise ValueError(f"Expected model output size 5 or 6, got {np.asarray(model_output).shape}.")

    is_prob = (not assume_logits) if assume_logits is not None else looks_like_probability_vector(raw)

    if is_prob:
        probs = raw.astype(np.float32)
    else:
        probs = softmax(raw, temperature=config.temperature)

    if probs.size == 5:
        probs6 = np.zeros(6, dtype=np.float32)
        probs6[1:] = probs
        probs = probs6

    probs = probs.astype(np.float32)
    probs /= max(float(np.sum(probs)), 1e-8)
    return probs


def entropy(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    return float(-np.sum(p * np.log(p + 1e-8)))


def top_two(probs: np.ndarray) -> Tuple[int, float, int, float, float]:
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    order = np.argsort(p)
    top_class    = int(order[-1])
    second_class = int(order[-2])
    top_prob     = float(p[top_class])
    second_prob  = float(p[second_class])
    return top_class, top_prob, second_class, second_prob, top_prob - second_prob


def valid_landmarks(landmarks: Optional[np.ndarray]) -> bool:
    if landmarks is None:
        return False
    lm = np.asarray(landmarks, dtype=np.float32)
    return (
        lm.ndim == 2
        and lm.shape[0] == 21
        and lm.shape[1] >= 2
        and not np.any(~np.isfinite(lm[:, :2]))
    )


def xy_landmarks(landmarks: np.ndarray) -> np.ndarray:
    return np.asarray(landmarks, dtype=np.float32)[:, :2]


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    bc = np.asarray(c, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    denom = max(float(np.linalg.norm(ba) * np.linalg.norm(bc)), 1e-8)
    cos_v = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_v)))


def palm_scale(lm: np.ndarray) -> float:
    return max(dist(lm[WRIST], lm[MIDDLE_MCP]), dist(lm[INDEX_MCP], lm[PINKY_MCP]), 1e-6)


def palm_center(lm: np.ndarray) -> np.ndarray:
    return np.mean(lm[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]], axis=0)


def finger_extended(lm: np.ndarray, finger: str, config: HeuristicConfig = DEFAULT_CONFIG) -> bool:
    s = palm_scale(lm)

    if finger == "thumb":
        _, _, ip, tip = FINGER_JOINTS["thumb"]
        center = palm_center(lm)
        return (
            dist(lm[tip], center) > dist(lm[ip], center) + config.finger_extension_extra * s
            and dist(lm[tip], lm[WRIST]) > dist(lm[ip], lm[WRIST]) + 0.02 * s
        )

    if finger not in LONG_FINGERS:
        raise ValueError(f"Unknown finger: {finger}")

    mcp, pip, dip, tip = FINGER_JOINTS[finger]
    straight = angle_degrees(lm[mcp], lm[pip], lm[dip]) > 145.0 and angle_degrees(lm[pip], lm[dip], lm[tip]) > 140.0
    extended = dist(lm[tip], lm[WRIST]) > dist(lm[pip], lm[WRIST]) + config.finger_extension_extra * s
    return bool(straight and extended)


def finger_states(lm: np.ndarray, config: HeuristicConfig = DEFAULT_CONFIG) -> Dict[str, bool]:
    return {f: finger_extended(lm, f, config) for f in ("thumb", "index", "middle", "ring", "pinky")}


def long_finger_extension_count(states: Dict[str, bool]) -> int:
    return sum(bool(states[f]) for f in LONG_FINGERS)


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

    thumb  = states["thumb"]
    index  = states["index"]
    middle = states["middle"]
    ring   = states["ring"]
    pinky  = states["pinky"]
    long_count = long_finger_extension_count(states)

    if pred_class == CLASS_FIST:
        return long_count <= 1

    if pred_class == CLASS_LIKE:
        return thumb and long_count <= 1

    if pred_class == CLASS_OK:
        return thumb_index_distance_ratio(lm) < config.ok_thumb_index_close

    if pred_class == CLASS_ONE:
        return index and long_count <= 2

    if pred_class == CLASS_PALM:
        spread = finger_spread_ratio(lm)
        if not (index and middle and ring and pinky):
            return False
        if spread > 0.70:
            return True
        return thumb and spread > config.palm_min_spread

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

    if second_class == CLASS_NA and margin < 0.10:
        debug["reason"] = f"second_is_NA_close_margin({margin:.3f})"
        return (CLASS_NA, debug) if return_debug else CLASS_NA

    if ent > config.max_entropy:
        debug["reason"] = f"high_entropy({ent:.3f} > {config.max_entropy:.3f})"
        return (CLASS_NA, debug) if return_debug else CLASS_NA

    if top_class == CLASS_PALM and ent > 1.00:
        debug["reason"] = f"palm_high_entropy({ent:.3f})"
        return (CLASS_NA, debug) if return_debug else CLASS_NA

    if config.use_landmark_rules:
        if not landmark_rule_pass(top_class, landmarks, config):
            debug["reason"] = "landmark_rule_failed"
            return (CLASS_NA, debug) if return_debug else CLASS_NA

    return (int(top_class), debug) if return_debug else int(top_class)


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def _load_model_once():
    global _MODEL

    if _MODEL is None:
        model = LandmarkOnlyModel(input_dim=42, num_classes=6)

        state_dict = torch.load(_MODEL_PATH, map_location=_DEVICE)
        state_dict = {k: v.float() if torch.is_floating_point(v) else v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(_DEVICE)
        model.eval()

        _MODEL = model

    return _MODEL


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

def _preprocess_landmarks(landmarks: np.ndarray, img_w: int, img_h: int) -> torch.Tensor:
    lm = np.asarray(landmarks, dtype=np.float32)

    if lm.ndim != 2 or lm.shape[0] != 21 or lm.shape[1] < 2:
        lm_xy = np.zeros((21, 2), dtype=np.float32)
    else:
        lm_xy = lm[:, :2].copy()
        lm_xy[:, 0] *= img_w
        lm_xy[:, 1] *= img_h
        wrist = lm_xy[0, :]
        lm_xy -= wrist
        max_dist = np.max(np.abs(lm_xy))
        if max_dist > 0:
            lm_xy /= max_dist

    return torch.tensor(lm_xy, dtype=torch.float32).unsqueeze(0).to(_DEVICE)


def _model_forward(cropped_img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    model = _load_model_once()
    img = np.asarray(cropped_img)
    img_h, img_w = img.shape[:2]
    lm_tensor = _preprocess_landmarks(landmarks, img_w, img_h)

    with torch.no_grad():
        logits = model(lm_tensor)

    return logits.squeeze(0).cpu().numpy()


# -----------------------------------------------------------------------------
# Scoring utility
# -----------------------------------------------------------------------------

def assignment_raw_score(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    score = 0
    for t, p in zip(y_true, y_pred):
        t, p = int(t), int(p)
        if t in VALID_CLASSES:
            if p == t:        score += 1
            elif p != CLASS_NA: score -= 2
        elif t == CLASS_NA:
            if p != CLASS_NA: score -= 2
        else:
            if p != CLASS_NA: score -= 2

    return int(score)


# ---------------------------------------------------------------------
# Required interface
# ---------------------------------------------------------------------

def _restore_landmark_pixel_coords(cropped_img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    lm = np.asarray(landmarks, dtype=np.float32).copy()

    if cropped_img is not None and cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
        h, w = cropped_img.shape[:2]
        lm[:, 0] *= w
        lm[:, 1] *= h

    return lm

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
    lm = np.asarray(landmarks, dtype=np.float32).copy()
    lm_pixel = _restore_landmark_pixel_coords(cropped_img, lm)

    model_output = _model_forward(cropped_img, lm)

    return int(final_decision(
        model_output=model_output,
        landmarks=lm_pixel,
        assume_logits=True,
    ))
