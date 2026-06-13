import random
import math
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
import cv2
cv2.setNumThreads(0)

class UltimateDataAugmentor:
    def __init__(self, 
                 p_blur=0.5,         # 高斯模糊的機率
                 p_color=0.5,        # 顏色抖動的機率
                 p_flip=0.5,         # 水平翻轉的機率
                 max_rotate=15.0,    # 最大旋轉角度 (正負 15 度)
                 max_scale=0.05,     # 縮放比例 (0.05 代表 0.95x ~ 1.05x)
                 max_translate=0.10,  # 平移比例 (0.03 代表 3% 的 bbox jitter)
                 p_landmark_jitter=0.5, # 骨架抖動機率
                 landmark_jitter_std=0.005, # 抖動的標準差 (0.005 代表 0.5% 的螢幕寬度誤差)
                 p_cutout=0.3,          # 隨機遮擋機率
                 cutout_size_pct=0.10,    # 遮擋方塊大小 (10% 的圖片寬高)
                 p_motion_blur=0.3,     # 觸發動態模糊的機率 (建議設 0.3~0.5)
                 motion_blur_size=15,    # 殘影的長度，必須是奇數 (15 約為快速揮手的殘影)
                 p_rotate_translate_scaling=0.5
                 ):
        self.p_blur = p_blur
        self.p_color = p_color
        self.p_flip = p_flip
        self.max_rotate = max_rotate
        self.max_scale = max_scale
        self.max_translate = max_translate
        self.p_landmark_jitter = p_landmark_jitter
        self.landmark_jitter_std = landmark_jitter_std
        self.p_cutout = p_cutout
        self.cutout_size_pct = cutout_size_pct
        self.p_motion_blur = p_motion_blur
        self.motion_blur_size = motion_blur_size if motion_blur_size % 2 == 1 else motion_blur_size + 1
        self.p_rotate_translate_scaling = p_rotate_translate_scaling

    def __call__(self, image: Image.Image, landmarks: np.ndarray):
        # 複製一份座標，避免改到原始資料
        landmarks = landmarks.copy()
        w, h = image.size

        # ==========================================
        # 純影像像素增強 (不動座標)
        # ==========================================
        if random.random() < self.p_blur:
            image = F.gaussian_blur(image, kernel_size=[11, 11], sigma=[0.5, 2.5])

        if random.random() < self.p_color:
            brightness_factor = random.uniform(0.5, 1.5)
            contrast_factor = random.uniform(0.5, 1.5)   
            image = F.adjust_brightness(image, brightness_factor)
            image = F.adjust_contrast(image, contrast_factor)

        # ==========================================
        # 動態模糊 (Motion Blur)
        # ==========================================
        if random.random() < self.p_motion_blur:
            img_cv = np.array(image)
            
            k_size = self.motion_blur_size
            kernel = np.zeros((k_size, k_size))
            kernel[k_size // 2, :] = 1.0
            
            angle = random.uniform(0, 360)
            M = cv2.getRotationMatrix2D((k_size / 2.0, k_size / 2.0), angle, 1)
            kernel = cv2.warpAffine(kernel, M, (k_size, k_size))
            kernel = kernel / np.sum(kernel)
            img_cv = cv2.filter2D(img_cv, -1, kernel)
            
            image = Image.fromarray(img_cv)

        # ==========================================
        # 隨機水平翻轉 (圖與點必須一起動)
        # ==========================================
        if random.random() < self.p_flip:
            image = F.hflip(image)
            landmarks[:, 0] = 1.0 - landmarks[:, 0]

        # ==========================================
        # 幾何同步增強 (旋轉、縮放、平移合一)
        # ==========================================
        if random.random() < self.p_rotate_translate_scaling:
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
        # 骨架獨立抖動 (Landmark Jitter)
        # ==========================================
        # 模擬 MediaPipe 偵測不準確的微小飄移 (只動座標，不動影像)
        if random.random() < self.p_landmark_jitter:
            # 產生與 landmarks 形狀相同的隨機常態分佈雜訊
            noise = np.random.normal(0, self.landmark_jitter_std, landmarks.shape)
            landmarks = landmarks + noise

        # ==========================================
        # 隨機遮擋 (Random Cutout)
        # ==========================================
        # 強迫模型不依賴單一手指特徵
        if random.random() < self.p_cutout:
            w, h = image.size
            cut_w, cut_h = int(w * self.cutout_size_pct), int(h * self.cutout_size_pct)
            
            # 隨機決定遮擋區塊的左上角座標
            x1 = random.randint(0, w - cut_w)
            y1 = random.randint(0, h - cut_h)
            
            # 使用 PyTorch 的 erase 畫上黑色方塊 (需將 PIL 轉 Tensor 再轉回)
            # 為了輕量，這裡直接用 PIL 的 ImageDraw
            from PIL import ImageDraw
            draw = ImageDraw.Draw(image)
            draw.rectangle([x1, y1, x1 + cut_w, y1 + cut_h], fill="black")

        # ==========================================
        # 安全邊界防護
        # ==========================================
        landmarks = np.clip(landmarks, 0.0, 1.0)

        return image, landmarks