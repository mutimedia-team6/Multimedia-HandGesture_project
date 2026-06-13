import numpy as np
from PIL import Image

def pad_to_square_and_resize(image: Image.Image, landmarks: np.ndarray, target_size: int = 224):
    """
    Proportionally scale the image, pad it with black borders to target_size x target_size, 
    and accurately transform the landmark coordinates.
    """
    w, h = image.size
    
    # 1. Calculate the proportional scale factor (based on the longest side)
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 2. Resize the image (maintaining aspect ratio without distortion)
    resized_image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # 3. Create a completely black square canvas
    square_image = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    
    # 4. Calculate the padding required to "center" the image
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    
    # 5. Paste the resized image onto the center of the black canvas
    square_image.paste(resized_image, (pad_x, pad_y))
    
    # ==========================================
    # 6. Coordinate adjustment: mathematically sync the landmarks
    # Logic: (original relative coordinate * original width/height * scale + padding offset) / new canvas size
    # ==========================================
    new_landmarks = landmarks.copy()
    new_landmarks[:, 0] = (landmarks[:, 0] * w * scale + pad_x) / target_size
    new_landmarks[:, 1] = (landmarks[:, 1] * h * scale + pad_y) / target_size
    
    # Safety guard to ensure coordinates do not go out of bounds
    new_landmarks = np.clip(new_landmarks, 0.0, 1.0)
    
    return square_image, new_landmarks