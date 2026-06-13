import random
import math
import numpy as np

class LandmarkOnlyAugmentor:
    def __init__(self, 
                 p_flip=0.5,         # 水平翻轉機率
                 max_rotate=15.0,    # 最大旋轉角度 (正負 15 度)
                 max_scale=0.05,     # 縮放比例 (0.05 代表 0.95x ~ 1.05x)
                 max_translate=0.03, # 平移比例 (相對於手部邊界框的 3%)
                 p_jitter=0.5,       # 骨架抖動機率 (模擬辨識誤差)
                 jitter_std=0.005):  # 抖動標準差
        
        self.p_flip = p_flip
        self.max_rotate = max_rotate
        self.max_scale = max_scale
        self.max_translate = max_translate
        self.p_jitter = p_jitter
        self.jitter_std = jitter_std

    def __call__(self, landmarks, w, h):
        """
        輸入:
            landmarks: shape (21, 2) 的 numpy array，為 0.0 ~ 1.0 的相對座標
            w: 原始圖片寬度
            h: 原始圖片高度
        輸出:
            經過資料增強後的「真實像素座標」 (21, 2)
        """
        # 複製一份避免污染原始資料
        new_landmarks = landmarks.copy()

        # ==========================================
        # 1. 隨機水平翻轉 (鏡像)
        # ==========================================
        if random.random() < self.p_flip:
            # 相對座標的翻轉非常簡單，直接用 1.0 減去 X 座標即可
            new_landmarks[:, 0] = 1.0 - new_landmarks[:, 0]

        # ==========================================
        # 2. 轉換為真實物理比例 (Pixel Space)
        # ⚠️ 關鍵：必須先轉回像素，再做旋轉，否則手勢會因為長寬比不同而變形！
        # ==========================================
        new_landmarks[:, 0] *= w
        new_landmarks[:, 1] *= h

        # 算出這隻手的中心點與寬高 (作為旋轉與平移的基準)
        center_x = np.mean(new_landmarks[:, 0])
        center_y = np.mean(new_landmarks[:, 1])
        bbox_w = np.max(new_landmarks[:, 0]) - np.min(new_landmarks[:, 0])
        bbox_h = np.max(new_landmarks[:, 1]) - np.min(new_landmarks[:, 1])

        # ==========================================
        # 3. 隨機旋轉 (Rotation) & 縮放 (Scaling)
        # ==========================================
        angle = random.uniform(-self.max_rotate, self.max_rotate)
        scale = random.uniform(1.0 - self.max_scale, 1.0 + self.max_scale)

        if angle != 0.0 or scale != 1.0:
            rad = math.radians(angle)
            cos_a = math.cos(rad) * scale
            sin_a = math.sin(rad) * scale

            # 平移到原點 -> 旋轉並縮放 -> 平移回原位
            new_landmarks[:, 0] -= center_x
            new_landmarks[:, 1] -= center_y

            # 矩陣旋轉公式
            rotated_x = new_landmarks[:, 0] * cos_a - new_landmarks[:, 1] * sin_a
            rotated_y = new_landmarks[:, 0] * sin_a + new_landmarks[:, 1] * cos_a

            new_landmarks[:, 0] = rotated_x + center_x
            new_landmarks[:, 1] = rotated_y + center_y

        # ==========================================
        # 4. 隨機平移 (Translation)
        # ==========================================
        tx = random.uniform(-self.max_translate, self.max_translate) * bbox_w
        ty = random.uniform(-self.max_translate, self.max_translate) * bbox_h
        new_landmarks[:, 0] += tx
        new_landmarks[:, 1] += ty

        # ==========================================
        # 5. 骨架獨立抖動 (Landmark Jitter)
        # ==========================================
        # 這是對純 Landmark 模型最重要的增強，能強迫模型不要死記硬背特定點位
        if random.random() < self.p_jitter:
            noise_x = np.random.normal(0, self.jitter_std * bbox_w, 21)
            noise_y = np.random.normal(0, self.jitter_std * bbox_h, 21)
            new_landmarks[:, 0] += noise_x
            new_landmarks[:, 1] += noise_y

        # 回傳增強過後的真實像素座標
        return new_landmarks