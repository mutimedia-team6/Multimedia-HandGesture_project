"""
heuristic.py

Conservative heuristic decision layer for the Hand Gesture Classification project.

Goal
----
The neural network gives a raw prediction, but because false triggers are penalized
more heavily than missed detections in this assignment, we do NOT blindly trust the
model. This file adds a conservative post-processing layer:

    model logits / probabilities
        -> confidence threshold
        -> probability margin threshold
        -> entropy threshold
        -> landmark geometry consistency check
        -> final class id in {0,1,2,3,4,5}

Class mapping required by the spec:
    0 = N/A
    1 = fist
    2 = like
    3 = ok
    4 = one
    5 = palm

Dependencies
------------
Only numpy is required. This is intentional because inference.py should stay simple
and lightweight for the official Google Colab evaluation environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np


# -----------------------------------------------------------------------------
# Class mapping
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# MediaPipe hand landmark indices
# -----------------------------------------------------------------------------
# MediaPipe Hands always uses the same 21 landmark indices:
# 0 wrist
# 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky.

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


# -----------------------------------------------------------------------------
# Tunable heuristic configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class HeuristicConfig:
    """
    All thresholds are centralized here so you can tune them later with a
    validation set.

    Important tuning idea:
    - If false triggers are too frequent, increase thresholds.
    - If too many valid gestures become N/A, decrease thresholds slightly.
    """

    # If the model outputs logits, we apply softmax(logits / temperature).
    # temperature > 1.0 makes probabilities less over-confident.
    temperature: float = 1.5

    # Entropy threshold. For 6 classes, maximum entropy is ln(6) ~= 1.79.
    # Higher entropy means the model is uncertain, so we reject as N/A.
    max_entropy: float = 1.35

    # Per-class minimum top probability.
    # OK and like are usually more dangerous because daily hand actions can look
    # similar to them, so their thresholds are slightly stricter by default.
    min_confidence: Dict[int, float] = None  # type: ignore[assignment]

    # Per-class minimum gap between the highest and second-highest probabilities.
    min_margin: Dict[int, float] = None  # type: ignore[assignment]

    # Whether to use landmark geometry checks.
    use_landmark_rules: bool = True

    # If True, if landmarks are missing or invalid, we reject valid gestures.
    reject_when_landmark_invalid: bool = True

    # Landmark-rule strictness. Increase these if false triggers are high.
    ok_thumb_index_close: float = 0.65
    palm_min_spread: float = 0.45
    finger_extension_extra: float = 0.05

    def __post_init__(self):
        # dataclass with frozen=True requires object.__setattr__.
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


# -----------------------------------------------------------------------------
# Basic math utilities
# -----------------------------------------------------------------------------

def _as_numpy_1d(x: Union[np.ndarray, list, tuple]) -> np.ndarray:
    """Convert model output to a flat float32 numpy array."""
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    return arr


def softmax(logits: Union[np.ndarray, list, tuple], temperature: float = 1.0) -> np.ndarray:
    """
    Numerically stable softmax.

    Args:
        logits: raw model outputs before softmax.
        temperature: softmax temperature. temperature > 1 reduces over-confidence.

    Returns:
        Probability vector that sums to 1.
    """
    z = _as_numpy_1d(logits)
    temperature = max(float(temperature), 1e-6)
    z = z / temperature
    z = z - np.max(z)  # numerical stability
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


def looks_like_probability_vector(x: np.ndarray) -> bool:
    """
    Guess whether a vector is already softmax probabilities.

    This lets final_decision() support both cases:
    - model returns logits
    - model already returns probabilities
    """
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
    """
    Convert model output to a 6-class probability vector.

    Supports two common model designs:
    1. 6-class output: [N/A, fist, like, ok, one, palm]
    2. 5-class output: [fist, like, ok, one, palm]
       In this case we prepend N/A probability as 0.0. Rejection is then handled
       by heuristics rather than by a learned N/A logit.

    Args:
        model_output: logits or probabilities from the model.
        config: heuristic configuration.
        assume_logits:
            - True: always apply softmax.
            - False: treat model_output as probabilities.
            - None: auto-detect.

    Returns:
        np.ndarray with shape (6,), sum ~= 1.
    """
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
        # Normalize defensively in case of tiny floating-point drift.
        probs = probs / max(float(np.sum(probs)), 1e-8)
    else:
        probs = softmax(raw, temperature=config.temperature)

    # If model predicts only five valid classes, add N/A as class 0.
    if probs.size == 5:
        probs6 = np.zeros(6, dtype=np.float32)
        probs6[1:] = probs
        probs = probs6

    # Final defensive normalization.
    probs = probs.astype(np.float32)
    probs = probs / max(float(np.sum(probs)), 1e-8)
    return probs


def entropy(probs: np.ndarray) -> float:
    """Shannon entropy of a probability vector."""
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    return float(-np.sum(p * np.log(p + 1e-8)))


def top_two(probs: np.ndarray) -> Tuple[int, float, int, float, float]:
    """
    Return top class, top prob, second class, second prob, and margin.
    """
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    order = np.argsort(p)
    top_class = int(order[-1])
    second_class = int(order[-2])
    top_prob = float(p[top_class])
    second_prob = float(p[second_class])
    margin = top_prob - second_prob
    return top_class, top_prob, second_class, second_prob, margin


# -----------------------------------------------------------------------------
# Landmark geometry utilities
# -----------------------------------------------------------------------------

def valid_landmarks(landmarks: Optional[np.ndarray]) -> bool:
    """
    Check whether landmarks look usable.

    Expected shape from the provided MediaPipe preprocessor is usually (21, 2):
    21 crop-relative points, each with x and y. We also accept (21, 3) and ignore z.
    """
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
    """Return only x,y coordinates as float32 array with shape (21, 2)."""
    return np.asarray(landmarks, dtype=np.float32)[:, :2]


def dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle ABC in degrees.

    Used to decide whether a finger joint is straight or bent.
    Large angle, e.g. 160 degrees, means straighter.
    Small angle means bent.
    """
    ba = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    bc = np.asarray(c, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    denom = max(float(np.linalg.norm(ba) * np.linalg.norm(bc)), 1e-8)
    cos_value = float(np.dot(ba, bc) / denom)
    cos_value = float(np.clip(cos_value, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_value)))


def palm_scale(lm: np.ndarray) -> float:
    """
    Estimate hand size for scale-invariant distance thresholds.

    We combine two stable distances:
    - wrist to middle MCP
    - index MCP to pinky MCP

    Using a scale prevents thresholds from depending on crop size.
    """
    wrist_to_middle = dist(lm[WRIST], lm[MIDDLE_MCP])
    index_to_pinky = dist(lm[INDEX_MCP], lm[PINKY_MCP])
    return max(wrist_to_middle, index_to_pinky, 1e-6)


def palm_center(lm: np.ndarray) -> np.ndarray:
    """
    Approximate palm center using wrist and MCP joints.
    """
    return np.mean(
        lm[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]], axis=0
    )


def finger_extended(lm: np.ndarray, finger: str, config: HeuristicConfig = DEFAULT_CONFIG) -> bool:
    """
    Roughly decide whether a finger is extended.

    For index/middle/ring/pinky:
    - A straight finger has a large angle around PIP.
    - Its tip is farther from the wrist than its PIP joint.

    For thumb:
    - Thumb geometry is different and strongly depends on left/right hand and camera angle.
    - We therefore use distance from palm center and wrist instead of y-direction rules.

    This is intentionally heuristic, not a replacement for the neural network.
    """
    s = palm_scale(lm)

    if finger == "thumb":
        _cmc, _mcp, ip, tip = FINGER_JOINTS["thumb"]
        center = palm_center(lm)

        # Thumb is likely extended if the thumb tip is clearly farther away from
        # the palm center than the thumb IP joint and also not too close to index MCP.
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

    # The exact angles vary across camera views. This is a balanced condition:
    # either both joints are quite straight, or the fingertip is clearly farther
    # from the wrist than the PIP joint.
    straight_by_angle = pip_angle > 145.0 and dip_angle > 140.0
    extended_by_distance = tip_from_wrist > pip_from_wrist + config.finger_extension_extra * s

    return bool(straight_by_angle and extended_by_distance)


def finger_states(lm: np.ndarray, config: HeuristicConfig = DEFAULT_CONFIG) -> Dict[str, bool]:
    """Return a dictionary saying whether each finger is extended."""
    return {
        "thumb": finger_extended(lm, "thumb", config),
        "index": finger_extended(lm, "index", config),
        "middle": finger_extended(lm, "middle", config),
        "ring": finger_extended(lm, "ring", config),
        "pinky": finger_extended(lm, "pinky", config),
    }


def long_finger_extension_count(states: Dict[str, bool]) -> int:
    """Number of extended non-thumb fingers."""
    return int(sum(bool(states[f]) for f in LONG_FINGERS))


def thumb_index_distance_ratio(lm: np.ndarray) -> float:
    """Distance between thumb tip and index tip, normalized by palm scale."""
    return dist(lm[THUMB_TIP], lm[INDEX_TIP]) / palm_scale(lm)


def finger_spread_ratio(lm: np.ndarray) -> float:
    """Index-tip to pinky-tip distance normalized by palm scale."""
    return dist(lm[INDEX_TIP], lm[PINKY_TIP]) / palm_scale(lm)


# -----------------------------------------------------------------------------
# Class-specific landmark rules
# -----------------------------------------------------------------------------

def landmark_rule_pass(
    pred_class: int,
    landmarks: Optional[np.ndarray],
    config: HeuristicConfig = DEFAULT_CONFIG,
) -> bool:
    """
    Check whether the landmark geometry is consistent with the predicted class.

    This function should be used as a rejection layer:
    - If the model says "ok" but thumb/index tips are not close -> reject as N/A.
    - If the model says "palm" but fingers are not extended -> reject as N/A.

    It should NOT be used as the only classifier, because real hand poses vary a lot.
    """
    if pred_class == CLASS_NA:
        return True

    if not valid_landmarks(landmarks):
        return not config.reject_when_landmark_invalid

    lm = xy_landmarks(landmarks)  # shape (21, 2)
    states = finger_states(lm, config)

    thumb = states["thumb"]
    index = states["index"]
    middle = states["middle"]
    ring = states["ring"]
    pinky = states["pinky"]

    long_count = long_finger_extension_count(states)

    if pred_class == CLASS_FIST:
        # Fist: long fingers should be folded. Thumb can be flexible because in
        # real images it may be outside, across the fingers, or partially hidden.
        return long_count <= 1

    if pred_class == CLASS_LIKE:
        # Like/thumbs-up: thumb should be extended, long fingers folded.
        # Allow at most one long finger to appear extended due to landmark noise.
        return thumb and long_count <= 1

    if pred_class == CLASS_OK:
        # OK: thumb tip and index tip should be close.
        close_ratio = thumb_index_distance_ratio(lm)
        return close_ratio < config.ok_thumb_index_close

    if pred_class == CLASS_ONE:
        # One: index extended, middle/ring/pinky folded. Thumb is flexible.
        return index and not middle and not ring and not pinky

    if pred_class == CLASS_PALM:
        # Palm: four long fingers extended and reasonably spread.
        spread = finger_spread_ratio(lm)
        return index and middle and ring and pinky and spread > config.palm_min_spread

    # Unknown class id should not be accepted.
    return False


# -----------------------------------------------------------------------------
# Final decision function
# -----------------------------------------------------------------------------

def final_decision(
    model_output: Union[np.ndarray, list, tuple],
    landmarks: Optional[np.ndarray],
    config: HeuristicConfig = DEFAULT_CONFIG,
    assume_logits: Optional[bool] = None,
    return_debug: bool = False,
):
    """
    Convert model output + landmarks into the final class id.

    Args:
        model_output:
            Either logits or probabilities.
            Shape can be (6,) for [N/A, fist, like, ok, one, palm]
            or (5,) for [fist, like, ok, one, palm].
        landmarks:
            MediaPipe crop-relative landmarks, usually shape (21, 2).
        config:
            Heuristic thresholds.
        assume_logits:
            True: force model_output to be treated as logits.
            False: force model_output to be treated as probabilities.
            None: auto-detect.
        return_debug:
            If True, return (decision, debug_dict).
            If False, return decision only.

    Returns:
        int in {0,1,2,3,4,5}, or (int, dict) when return_debug=True.
    """
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

    # If model already predicts N/A, keep N/A. This is the safest behavior for
    # false-trigger prevention.
    if top_class == CLASS_NA:
        debug["reason"] = "model_top_is_NA"
        return (CLASS_NA, debug) if return_debug else CLASS_NA

    # Reject invalid class ids defensively.
    if top_class not in VALID_CLASSES:
        debug["reason"] = "invalid_top_class"
        return (CLASS_NA, debug) if return_debug else CLASS_NA

    # 1) Confidence threshold.
    min_conf = float(config.min_confidence.get(top_class, 0.75))
    if top_prob < min_conf:
        debug["reason"] = f"low_confidence({top_prob:.3f} < {min_conf:.3f})"
        return (CLASS_NA, debug) if return_debug else CLASS_NA

    # 2) Probability margin threshold.
    min_margin = float(config.min_margin.get(top_class, 0.15))
    if margin < min_margin:
        debug["reason"] = f"low_margin({margin:.3f} < {min_margin:.3f})"
        return (CLASS_NA, debug) if return_debug else CLASS_NA

    # 3) Entropy threshold.
    if ent > config.max_entropy:
        debug["reason"] = f"high_entropy({ent:.3f} > {config.max_entropy:.3f})"
        return (CLASS_NA, debug) if return_debug else CLASS_NA

    # 4) Landmark geometry consistency check.
    if config.use_landmark_rules:
        if not landmark_rule_pass(top_class, landmarks, config):
            debug["reason"] = "landmark_rule_failed"
            return (CLASS_NA, debug) if return_debug else CLASS_NA

    return (int(top_class), debug) if return_debug else int(top_class)


# Alias name if your inference.py prefers this wording.
def apply_heuristics(
    model_output: Union[np.ndarray, list, tuple],
    landmarks: Optional[np.ndarray],
    config: HeuristicConfig = DEFAULT_CONFIG,
    assume_logits: Optional[bool] = None,
    return_debug: bool = False,
):
    """Alias for final_decision()."""
    return final_decision(
        model_output=model_output,
        landmarks=landmarks,
        config=config,
        assume_logits=assume_logits,
        return_debug=return_debug,
    )


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
