# Hand Gesture Classification Inference

## Overview

This submission implements a compact hand gesture classifier for six classes:

| Class ID | Meaning |
| -------: | ------- |
|        0 | N/A     |
|        1 | fist    |
|        2 | like    |
|        3 | ok      |
|        4 | one     |
|        5 | palm    |

The classifier takes two inputs:

1. `cropped_img`: an RGB cropped hand image as a NumPy array.
2. `landmarks`: 21 hand landmarks as a NumPy array.

The model uses a MobileNetV3-small image branch and a landmark MLP branch. The two feature vectors are concatenated and passed through a final classifier.

## File Structure

The submitted zip file should follow this structure:

```text
team_X.zip
├── inference.py
├── model/
│   └── fusion_mobilenetv3_landmark_best.pth
├── requirements.txt
└── README.md
```

## Environment

This inference code is intended to run in a fresh Google Colab runtime.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

The evaluator should call the `predict()` function in `inference.py`:

```python
from inference import predict

pred = predict(cropped_img, landmarks)
```

The return value is an integer class ID:

```text
0 = N/A
1 = fist
2 = like
3 = ok
4 = one
5 = palm
```

## Notes

* The model checkpoint is loaded from `model/fusion_mobilenetv3_landmark_best.pth`.
* All paths are relative to `inference.py`.
* The inference code runs on CPU by default for compatibility with the official evaluation environment.
* ImageNet weights are not downloaded during inference; the submitted checkpoint already contains the trained model weights.

