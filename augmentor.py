import random
import math
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

class UltimateDataAugmentor:
    def __init__(self, 
                 p_blur=0.5,         # 高斯模糊的機率
                 p_color=0.5,        # 顏色抖動的機率
                 p_flip=0.5,         # 水平翻轉的機率
                 max_rotate=15.0,    # 最大旋轉角度 (正負 15 度)
                 max_scale=0.05,     # 縮放比例 (0.05 代表 0.95x ~ 1.05x)
                 max_translate=0.03  # 平移比例 (0.03 代表 3% 的 bbox jitter)
                 ):
        self.p_blur = p_blur
        self.p_color = p_color
        self.p_flip = p_flip
        self.max_rotate = max_rotate
        self.max_scale = max_scale
        self.max_translate = max_translate

    def __call__(self, image: Image.Image, landmarks: np.ndarray):
        # 複製一份座標，避免改到原始資料
        landmarks = landmarks.copy()
        w, h = image.size

        # ==========================================
        # 第一關：純影像像素增強 (不動座標)
        # ==========================================
        if random.random() < self.p_blur:
            image = F.gaussian_blur(image, kernel_size=[11, 11], sigma=[0.5, 2.5])

        if random.random() < self.p_color:
            brightness_factor = random.uniform(0.5, 1.5)
            contrast_factor = random.uniform(0.5, 1.5)   
            image = F.adjust_brightness(image, brightness_factor)
            image = F.adjust_contrast(image, contrast_factor)

        # ==========================================
        # 第二關：隨機水平翻轉 (圖與點必須一起動)
        # ==========================================
        if random.random() < self.p_flip:
            image = F.hflip(image)
            landmarks[:, 0] = 1.0 - landmarks[:, 0]

        # ==========================================
        # 第三關：幾何同步增強 (旋轉、縮放、平移合一)
        # ==========================================
        # 1. 隨機生成本次要變換的參數
        angle = random.uniform(-self.max_rotate, self.max_rotate) if self.max_rotate > 0 else 0
        scale_factor = random.uniform(1.0 - self.max_scale, 1.0 + self.max_scale) if self.max_scale > 0 else 1.0
        tx_pct = random.uniform(-self.max_translate, self.max_translate) if self.max_translate > 0 else 0.0
        ty_pct = random.uniform(-self.max_translate, self.max_translate) if self.max_translate > 0 else 0.0

        # 如果有任何幾何擾動，就啟動同步變換
        if angle != 0 or scale_factor != 1.0 or tx_pct != 0.0 or ty_pct != 0.0:
            
            # --- 【圖片端】使用 F.affine 一次到位 ---
            # F.affine 的平移參數需要傳入真實的「像素移動量」
            tx_pixels = int(tx_pct * w)
            ty_pixels = int(ty_pct * h)
            
            # fill=0 代表移動或旋轉後暴露出的圖像邊界，用黑色補齊
            image = F.affine(
                image, 
                angle=angle, 
                translate=[tx_pixels, ty_pixels], 
                scale=scale_factor, 
                shear=0, 
                fill=0
            )
            
            # --- 【座標端】像素級數學矩陣同步變換 ---
            # 步驟 A: 將 0~1 的相對座標轉換為「真實像素座標」
            pixel_landmarks = landmarks * np.array([w, h])
            # pixel_landmarks = (landmarks * np.array([w, h])).astype(int)
            
            # 定義旋轉中心為圖片的正中央 (像素座標)
            cx, cy = w // 2, h // 2
            
            # 步驟 B: 旋轉與縮放矩陣運算
            # 影像 Y 軸朝下的座標系中，逆時針旋轉對應的標準二維旋轉矩陣
            theta = math.radians(-angle)
            c, s = math.cos(theta), math.sin(theta)
            R = np.array([[c,  s], 
                          [-s, c]])
            
            # 1. 將座標原點移到圖片中心
            pixel_landmarks[:, 0] -= cx
            pixel_landmarks[:, 1] -= cy
            
            # 2. 矩陣相乘套用旋轉，並乘上縮放比例
            pixel_landmarks = np.dot(pixel_landmarks, R.T) * scale_factor
            
            # 3. 移回原本中心，並外加「平移的像素量」
            pixel_landmarks[:, 0] += (cx + tx_pixels)
            pixel_landmarks[:, 1] += (cy + ty_pixels)
            
            # 步驟 C: 重新轉回 0~1 的相對比例
            landmarks = pixel_landmarks / np.array([w, h])

        # ==========================================
        # 第四關：安全邊界防護
        # ==========================================
        # 確保經過平移、旋轉、縮放後，點不會因為浮點數誤差跑出 0.0 ~ 1.0 的範圍
        landmarks = np.clip(landmarks, 0.0, 1.0)

        return image, landmarks