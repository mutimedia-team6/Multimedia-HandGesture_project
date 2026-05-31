import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 載入你的模型推論函數與類別名稱
from inference import predict, CLASS_NAMES

def main():
    # ==========================================
    # 1. 設定資料夾路徑 (請替換成你的實際路徑)
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
    print(f"🚀 開始極速評估模型，預計測試總量: {total_images} 張圖片...")
    
    # 紀錄開始時間
    start_time = time.time()
    
    # ==========================================
    # 2. 迴圈跑遍每一張測試圖片
    # ==========================================
    # tqdm 會自動幫你計算並顯示剩餘時間 (ETA)
    for index, row in tqdm(df.iterrows(), total=total_images, desc="Evaluating"):
        img_filename = row['idx'] 

        if not img_filename.endswith(('.jpg', '.png', '.jpeg')):
            img_filename += ".jpg"

        true_label_str = row['label_name']
        
        img_path = CROP_DIR / f"{img_filename}"
        base_name = os.path.splitext(img_filename)[0]
        lmk_path = LMK_DIR / f"{base_name}.npy"

        if not img_path.exists():
            print(f"⚠️ 找不到圖片: {img_path}")
            continue 
        if not lmk_path.exists():
            print(f"⚠️ 找不到座標: {lmk_path}")
            continue
        true_class_id = label_map.get(true_label_str, 0)

        # 讀圖與座標
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        try:
            landmarks = np.load(str(lmk_path))
            
            # 模型預測
            pred_class_id = predict(img_rgb, landmarks)
            
            y_true.append(true_class_id)
            y_pred.append(pred_class_id)
        except Exception as e:
            # 不要再 pass 了，把它印出來，並且直接中斷程式！
            print(f"\n❌ 抓到真兇了！在預測時發生錯誤: {e}")
            break

    # 計算總花費時間
    elapsed_time = time.time() - start_time
    valid_tests = len(y_true)

    # ==========================================
    # 3. 計算 Accuracy 與畫出 Truth Table
    # ==========================================
    print("\n" + "="*50)
    print("📊 測試結果總結 (Test Summary)")
    print("="*50)
    
    # 算總體準確率 (Accuracy)
    acc = accuracy_score(y_true, y_pred)
    print(f"✅ 有效測試數量 : {valid_tests} 張")
    print(f"⏱️ 總耗時       : {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分鐘)")
    print(f"⚡ 平均推論速度 : {(elapsed_time/valid_tests)*1000:.2f} 毫秒/張 (ms/iter)")
    print(f"🏆 總體準確率 (Accuracy): {acc*100:.2f} %")
    print("="*50)
    
    # 畫混淆矩陣
    labels = [0, 1, 2, 3, 4, 5]
    display_names = [CLASS_NAMES[i] for i in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(9, 7))
    sns.set_theme(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=display_names, 
                yticklabels=display_names)
    
    plt.xlabel('Predicted Label (模型預測)')
    plt.ylabel('True Label (真實答案)')
    plt.title(f'Confusion Matrix (Accuracy: {acc*100:.1f}%)')
    plt.tight_layout()

    # ... (前面算 accuracy 跟 seaborn 畫 heatmap 的部分不動) ...
    
    plt.xlabel('Predicted Label (模型預測)')
    plt.ylabel('True Label (真實答案)')
    plt.title(f'Confusion Matrix (Accuracy: {acc*100:.1f}%)')
    plt.tight_layout()

    # ==========================================
    # 🌟 新增：自動存檔功能 (加上時間戳記避免覆蓋)
    # ==========================================
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # 取得當下時間
    
    # 1. 儲存圖片 (這行一定要寫在 plt.show() 前面！)
    plot_filename = f"confusion_matrix_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\n📁 [存檔成功] 混淆矩陣圖表已儲存至: {plot_filename}")

    # 2. 儲存文字測試報告
    report_filename = f"test_report_{timestamp}.txt"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write(f"測試時間: {datetime.datetime.now()}\n")
        f.write("="*50 + "\n")
        f.write(f"✅ 有效測試數量 : {valid_tests} 張\n")
        f.write(f"⏱️ 總耗時       : {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分鐘)\n")
        f.write(f"🏆 總體準確率   : {acc*100:.2f} %\n")
        f.write("="*50 + "\n")
        f.write("詳細分類報告 (Classification Report):\n\n")
        f.write(classification_report(y_true, y_pred, target_names=display_names, labels=labels))
    
    print(f"📁 [存檔成功] 文字測試報告已儲存至: {report_filename}\n")
    # ==========================================

    # 最後再顯示視窗讓你看
    plt.show()

if __name__ == "__main__":
    main()