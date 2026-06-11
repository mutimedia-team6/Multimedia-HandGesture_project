import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch

# 最標準的寫法
import mediapipe as mp  # type: ignore[import-not-found]

from inference import predict, CLASS_NAMES

# 正常初始化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# ... 下面的 process_image 函數內容保持不變 ...
# (注意：底下的繪製函數 mp_drawing.draw_landmarks(...) 跟 mp_hands.HAND_CONNECTIONS 都可以直接繼續使用)

def process_image(image_np):
    """
    處理影像、擷取 Landmark 並進行預測
    """
    # MediaPipe 需要 RGB 格式
    h, w, _ = image_np.shape
    results = hands.process(image_np)

    if not results.multi_hand_landmarks:
        return image_np, "N/A (未偵測到手部)"

    # 針對偵測到的第一隻手
    hand_landmarks = results.multi_hand_landmarks[0]
    
    # 將 MediaPipe 的正規化座標轉換為你模型所需的絕對像素座標 (21, 2)
    lm_array = np.zeros((21, 2), dtype=np.float32)
    for i, lm in enumerate(hand_landmarks.landmark):
        lm_array[i] = [lm.x * w, lm.y * h]

    # 呼叫你的預測函數
    pred_class = predict(image_np, lm_array)
    pred_name = CLASS_NAMES.get(pred_class, "Unknown")

    # 在影像上繪製骨架與預測結果
    annotated_image = image_np.copy()
    mp_drawing.draw_landmarks(
        annotated_image, 
        hand_landmarks, 
        mp_hands.HAND_CONNECTIONS
    )
    
    # 加上文字標籤
    cv2.putText(
        annotated_image, 
        f"Gesture: {pred_name}", 
        (10, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1.5, 
        (0, 255, 0), 
        3
    )

    return annotated_image, pred_name

# --- Streamlit 網頁介面設計 ---
st.set_page_config(page_title="手勢辨識系統", layout="centered")

st.title("✋ 深度學習手勢辨識展示")
st.markdown("上傳圖片或是使用網路攝影機拍一張照，模型會自動擷取骨架並判斷手勢！")

# 提供兩種輸入方式
source_option = st.radio("選擇影像來源：", ("使用攝影機拍照", "上傳圖片"))

image_file = None
if source_option == "使用攝影機拍照":
    image_file = st.camera_input("拍張照吧")
else:
    image_file = st.file_uploader("上傳一張包含手勢的圖片 (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if image_file is not None:
    # 將上傳的檔案轉換為 OpenCV/Numpy 格式
    image = Image.open(image_file).convert('RGB')
    image_np = np.array(image)

    with st.spinner('正在分析手部骨架與推論中...'):
        # 進行處理與預測
        processed_img, gesture_name = process_image(image_np)

    st.success(f"**預測結果：{gesture_name}**")
    
    # 顯示畫上骨架與標籤的圖片
    st.image(processed_img, caption="分析結果", use_container_width=True)