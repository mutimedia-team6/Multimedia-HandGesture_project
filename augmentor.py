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
        # 初始化所有參數，這樣隊友以後要調參數只要改這裡就好
        self.p_blur = p_blur
        self.p_color = p_color
        self.p_flip = p_flip
        self.max_rotate = max_rotate
        self.max_scale = max_scale
        self.max_translate = max_translate

    def __call__(self, image: Image.Image, landmarks: np.ndarray):
        """
        傳入:
            image: PIL.Image 格式的 512x512 手部裁切圖
            landmarks: numpy array 格式的座標，形狀為 (21, 2)，數值介於 0.0 ~ 1.0 之間
        回傳:
            增強後的 image, 增強後的 landmarks
        """
        # 為了安全起見，先複製一份座標，避免改到原始資料
        landmarks = landmarks.copy()

        # ==========================================
        # 第一關：純影像像素增強 (不動座標)
        # ==========================================
        
        # 1. 隨機高斯模糊 (破解助教 Blur 考題)
        if random.random() < self.p_blur:
            image = F.gaussian_blur(image, kernel_size=[11, 11], sigma=[0.5, 2.5])

        # 2. 隨機顏色抖動 (增強真實環境光線魯棒性)
        if random.random() < self.p_color:
            brightness_factor = random.uniform(0.5, 1.5) # 變暗到變亮
            contrast_factor = random.uniform(0.5, 1.5)   # 對比度增減
            image = F.adjust_brightness(image, brightness_factor)
            image = F.adjust_contrast(image, contrast_factor)

        # ==========================================
        # 第二關：幾何同步增強 (圖與點必須一起動！)
        # ==========================================

        # 3. 隨機水平翻轉 (擴充左右手資料)
        if random.random() < self.p_flip:
            image = F.hflip(image)
            # 圖片水平翻轉了，Landmark 的 X 座標也要跟著翻轉 (1 - X)
            landmarks[:, 0] = 1.0 - landmarks[:, 0]

        # 4. 隨機旋轉與縮放 (合在同一個矩陣運算最快)
        angle = random.uniform(-self.max_rotate, self.max_rotate) if self.max_rotate > 0 else 0
        scale_factor = random.uniform(1.0 - self.max_scale, 1.0 + self.max_scale) if self.max_scale > 0 else 1.0

        if angle != 0 or scale_factor != 1.0:
            # 旋轉圖片
            if angle != 0:
                image = F.rotate(image, angle)
            
            # 旋轉與縮放座標點 (以圖片中心 0.5, 0.5 為軸心)
            # PyTorch 的 rotate 是逆時針，但影像 Y 軸朝下，為了精準對齊，數學公式如下：
            theta = math.radians(-angle) # 轉換為弧度
            c, s = math.cos(theta), math.sin(theta)
            R = np.array([[c, -s], 
                          [s,  c]])
            
            # 步驟：移到原點 -> 旋轉並縮放 -> 移回中心
            landmarks = landmarks - 0.5
            landmarks = np.dot(landmarks, R.T) * scale_factor
            landmarks = landmarks + 0.5

        # ==========================================
        # 第三關：純座標擾動增強 (破解 Bbox Jitter)
        # ==========================================

        # 5. 隨機平移 (圖片不動，只動座標，模擬裁切框歪掉)
        if self.max_translate > 0:
            tx = random.uniform(-self.max_translate, self.max_translate)
            ty = random.uniform(-self.max_translate, self.max_translate)
            
            landmarks[:, 0] += tx
            landmarks[:, 1] += ty
            
            # 避免平移後點跑出 0~1 的範圍（可選，但建議加上）
            landmarks = np.clip(landmarks, 0.0, 1.0)

        return image, landmarks