import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from pathlib import Path
from tqdm import tqdm
from PIL import Image  # 🌟 新增：為了配合你的 Augmentor，需要載入 PIL
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 載入你的模型推論函數與類別名稱
from inference import predict, CLASS_NAMES

# 🌟 新增：載入你寫好的資料增強器 (假設你的檔案叫做 augmentor.py)
try:
    from augmentor import UltimateDataAugmentor
except ImportError:
    print("⚠️ 警告：找不到 augmentor.py 或 DataAugmentor 類別。若不開啟增強測試則無妨。")

def main():
    # ==========================================
    # 🌟 測試設定區 (Test Configuration)
    # ==========================================
    # 開啟這個開關，就會對每一張測試圖片與座標「加料」
    SIMULATE_NOISE = True
    
    if SIMULATE_NOISE:
        print("🌪️ [警告] 已啟動『抗噪模擬測試模式』！")
        # 建立增強器：這裡的機率設高一點，確保測試時真的有加到雜訊
        # 注意：測試時建議 p_flip=0 (不要翻轉)，除非你的模型原本就有練雙手鏡像
        noise_generator = UltimateDataAugmentor(
            p_blur=0.7,         # 70% 機率畫面模糊 (模擬手部動態殘影)
            p_color=0.7,        # 70% 機率亮度/對比改變 (模擬不同光源)
            p_flip=0.0,         # 0% 測試集不隨機左右翻轉
            max_rotate=10.0,    # 輕微旋轉
            max_scale=0.05,     # 輕微縮放 (模擬遠近)
            max_translate=0.05  # 🌟 最重要的 Bbox Jitter (座標偏移)
        )
    else:
        print("平靜模式：讀取原始乾淨圖片進行測試。")

    # ==========================================
    # 1. 設定資料夾路徑
    # ==========================================
    BASE_DIR = Path("/mnt/d/hagrid_project/dataset_v1_processed_detected")
    CROP_DIR = BASE_DIR / "crops"
    LMK_DIR = BASE_DIR / "landmarks"
    CSV_PATH = BASE_DIR / "labels_fixed.csv"

    if not CSV_PATH.exists():
        print(f"找不到標籤檔案: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    label_map = {"N_A": 0, "fist": 1, "like": 2, "ok": 3, "one": 4, "palm": 5}
    
    y_true = [] 
    y_pred = [] 

    total_images = len(df)
    print(f"🚀 開始評估模型，預計測試總量: {total_images} 張圖片...")
    
    start_time = time.time()
    
    # ==========================================
    # 2. 迴圈跑遍每一張測試圖片
    # ==========================================
    for index, row in tqdm(df.iterrows(), total=total_images, desc="Evaluating"):
        img_filename = str(row['idx']) 

        if not img_filename.endswith(('.jpg', '.png', '.jpeg')):
            img_filename += ".jpg"

        true_label_str = row['label_name']
        img_path = CROP_DIR / f"{img_filename}"
        base_name = os.path.splitext(img_filename)[0]
        lmk_path = LMK_DIR / f"{base_name}.npy"

        if not img_path.exists() or not lmk_path.exists():
            continue 

        true_class_id = label_map.get(true_label_str, 0)

        # 讀圖與座標
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        try:
            landmarks = np.load(str(lmk_path))
            
            # ==========================================
            # 🌟 核心：在此處注入雜訊與干擾
            # ==========================================
            if SIMULATE_NOISE:
                # 把 numpy array 轉成 PIL Image 給 Augmentor 吃
                img_pil = Image.fromarray(img_rgb)
                
                # 執行增強 (圖片變糊/變暗、座標集體位移)
                img_pil, landmarks = noise_generator(img_pil, landmarks)
                
                # 轉回 numpy array 準備餵給 predict 模型
                img_rgb = np.array(img_pil)
            # ==========================================

            # 模型預測
            pred_class_id = predict(img_rgb, landmarks)
            
            y_true.append(true_class_id)
            y_pred.append(pred_class_id)

        except Exception as e:
            print(f"\n❌ 抓到真兇了！在預測時發生錯誤: {e}")
            break

    elapsed_time = time.time() - start_time
    valid_tests = len(y_true)

    if valid_tests == 0:
        print("沒有成功預測任何圖片，結束程式。")
        return

    # ==========================================
    # 3. 計算 Accuracy 與畫出 Truth Table
    # ==========================================
    print("\n" + "="*50)
    print("📊 測試結果總結 (Test Summary)")
    print("="*50)
    
    acc = accuracy_score(y_true, y_pred)
    print(f"✅ 有效測試數量 : {valid_tests} 張")
    print(f"⏱️ 總耗時       : {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分鐘)")
    print(f"⚡ 平均推論速度 : {(elapsed_time/valid_tests)*1000:.2f} 毫秒/張 (ms/iter)")
    print(f"🏆 總體準確率   : {acc*100:.2f} %")
    print("="*50)
    
    labels = [0, 1, 2, 3, 4, 5]
    display_names = [CLASS_NAMES[i] for i in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(9, 7))
    sns.set_theme(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Oranges' if SIMULATE_NOISE else 'Blues', 
                xticklabels=display_names, 
                yticklabels=display_names)
    
    # 標題自動標示是否為加噪測試
    title_prefix = "[Noisy Test] " if SIMULATE_NOISE else ""
    plt.xlabel('Predicted Label (模型預測)')
    plt.ylabel('True Label (真實答案)')
    plt.title(f'{title_prefix}Confusion Matrix (Accuracy: {acc*100:.1f}%)')
    plt.tight_layout()

    # ==========================================
    # 🌟 自動存檔功能
    # ==========================================
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "noisy" if SIMULATE_NOISE else "clean"
    
    plot_filename = f"confusion_matrix_{mode_str}_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\n📁 [存檔成功] 混淆矩陣圖表已儲存至: {plot_filename}")

    report_filename = f"test_report_{mode_str}_{timestamp}.txt"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write(f"測試時間: {datetime.datetime.now()}\n")
        f.write(f"測試模式: {'抗噪干擾測試 (Noisy)' if SIMULATE_NOISE else '乾淨原圖測試 (Clean)'}\n")
        f.write("="*50 + "\n")
        f.write(f"✅ 有效測試數量 : {valid_tests} 張\n")
        f.write(f"🏆 總體準確率   : {acc*100:.2f} %\n")
        f.write("="*50 + "\n")
        f.write(classification_report(y_true, y_pred, target_names=display_names, labels=labels))
    
    print(f"📁 [存檔成功] 文字測試報告已儲存至: {report_filename}\n")
    plt.show()

if __name__ == "__main__":
    main()