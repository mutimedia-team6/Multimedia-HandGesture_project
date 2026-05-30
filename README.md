# Hand Gesture Classification on Edge Devices

This project focuses on building a hand gesture classification system that can run efficiently on edge devices. The system combines image-based visual features and hand landmark features to classify hand gestures while keeping the model lightweight enough for deployment.

## Team

- 黃湘晴
- 韋妍伃
- 陳穎達
- 游宇宸

## Project Goal

The goal is to classify hand gestures using a compact deep learning model suitable for edge-device deployment. The system is designed to balance classification accuracy, model size, and inference efficiency.

## System Pipeline

The proposed pipeline contains three main parts:

1. Data augmentation
2. Model training
3. Heuristic-based uncertainty handling

## Data Augmentation

### Image-Stream Augmentation

The image input stream uses augmentation methods to improve robustness:

- Gaussian blur
- Color jitter
  - Brightness adjustment
  - Contrast adjustment
  - Saturation adjustment

### Landmark-Stream Augmentation

The landmark input stream uses augmentation on the 21 hand joint coordinates:

- Random translation: globally shifts all 21 landmark coordinates by a small random offset.
- Random scaling: uniformly scales the landmark coordinates relative to the wrist base point.

## Model Training

The system uses both image features and landmark features.

### Image Processing

Two training strategies will be evaluated, and the better-performing method will be selected as the final image model.

- Lightweight ImageNet-pretrained model
  - Use transfer learning and fine-tuning on a small pretrained model.
- Large model followed by compression
  - Train a larger teacher model, then distill it into a smaller student model.

### Landmark Processing

The 21 hand landmarks are fed into a multilayer perceptron (MLP) to extract landmark-based features.

### Classification

The feature vectors from the image model and the landmark MLP are concatenated. A fully connected classifier is then applied with focal loss, and Softmax is used to output 6-class probabilities.

## Heuristic Rules

To improve reliability, the system applies heuristic checks before returning a final prediction.

1. Confidence thresholding
   - Return `N/A` when the model prediction confidence is too low.
2. Probability margin check
   - Return `N/A` when the top prediction is too close to the second-best prediction.
3. Entropy thresholding
   - Return `N/A` when the Softmax output entropy is too high, indicating uncertainty across all classes.
4. Landmark-based gesture rules
   - Return `N/A` when the predicted gesture does not match the expected hand landmark geometry.

## Research Methodology

### 1. Literature Review and Baseline Analysis

We will conduct a literature review to analyze existing approaches and prior research. This phase aims to understand current solutions and establish a theoretical baseline for the system.

### 2. Concurrent Development and Task Allocation

To accelerate development, the team will divide the workload and execute three core tasks in parallel:

- Data augmentation
- Model training
- Heuristic algorithm design

### 3. System Integration and Model Optimization

The parallel work streams will be integrated for final testing and fine-tuning. The main objective is to maximize model accuracy while minimizing the model footprint.

## Schedule

| Date | Milestone |
| --- | --- |
| 6/2 | Finish individual parts |
| 6/6 | Finish first-stage system integration and model optimization |
| 6/11 | Finish final presentation and final model |

## Dataset Google Drive
https://drive.google.com/drive/folders/1qHkqg_47XUhlyBu4aE3nqrSZnt-Ah5UC?usp=drive_link
- hagrid_set_v1_medium: Totally 10.000images withe 5-target class: 1,000, 29 ono-target class: each 172, total 5,000.
- hagrid_set_v3: Totally 100.000images withe 5-target class: 10,000, 29 ono-target class: each 1,724, total 50,000.
