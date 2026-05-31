import zipfile
import io
import os
import random
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import pandas as pd  # 💡 新增：為了最後產出 labels.csv

# 正確引入助教給的 hand_preprocess.py 裡面的類別
from hand_preprocess import MediaPipeHandPreprocessor 

# ==================== 🛠️ 10000張滑塊控制面板 ====================
ZIP_FILE_PATH = Path(r"/mnt/d/hagrid_project/hagridv2_512.zip")      

# 💡 這次輸出後的資料夾，就是直接給 PyTorch 吃的終極版
OUTPUT_DIR = Path(r"/mnt/d/hagrid_project/dataset_v2_processed")     
START_INDEX = 0                                                            

# 5 個專題指定的標準目標手勢
TARGET_CATEGORIES = {'fist', 'like', 'ok', 'one', 'palm'}

# 💡 標籤對照表 (0~5)，給 CSV 紀錄用的
LABEL_MAP = {
    "N_A": 0,
    "fist": 1,
    "like": 2,
    "ok": 3,
    "one": 4,
    "palm": 5,
}

# 🎯 【10,000 張的黃金配額比例】
MAX_PER_TARGET = 10000    # 每個目標類別 1000 張
MAX_PER_NOISE = 1724       # 29個雜訊每個 172 張

RANDOM_SEED = 0          # 固定隨機種子碼，確保全局打亂順序一致
# ===============================================================

def main():
    if not ZIP_FILE_PATH.exists():
        print(f"❌ 找不到原始壓縮檔，請檢查路徑：{ZIP_FILE_PATH}")
        return

    preprocessor = MediaPipeHandPreprocessor()
    
    # 💡 建立扁平化的資料夾結構 (不再分 fist/, ok/ 了，全部倒進這兩個)
    crop_dir = OUTPUT_DIR / "crops"
    landmark_dir = OUTPUT_DIR / "landmarks"
    crop_dir.mkdir(parents=True, exist_ok=True)
    landmark_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 [一條龍終極流水線] 目錄：{OUTPUT_DIR} | 起始點控制：{START_INDEX}")
    
    saved_counters = {}
    records = [] # 💡 用來記錄每一筆資料，最後轉成 CSV

    with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as z:
        print("🔍 步驟一：正在建立壓縮檔全量索引...")
        all_files = z.namelist()
        
        # 👑 多層路徑智慧探測
        raw_img_files = []
        for f in all_files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and ".ipynb_checkpoints" not in f:
                parts = f.split('/')
                
                matched_cat = None
                for part in reversed(parts):
                    if part and not part.endswith(('.jpg', '.jpeg', '.png')) and part != "hagridv2_512":
                        matched_cat = part
                        break
                
                if matched_cat:
                    raw_img_files.append((f, matched_cat))

        # 動態抓取所有真正的手勢類別來初始化計數器
        all_detected_cats = set(cat_name for _, cat_name in raw_img_files)
        for cat in all_detected_cats:
            saved_counters[cat] = 0

        print(f"📂 成功辨識出 {len(all_detected_cats)} 個手勢類別。原始總圖數：{len(raw_img_files)}")
        
        # 🔥 固定 Seed 全局場景大打亂
        print(f"🎴 正在注入 Seed {RANDOM_SEED} 進行全局打亂...")
        random.seed(RANDOM_SEED)
        random.shuffle(raw_img_files)
        
        # 👑 發動滑塊切片
        sliced_img_files = raw_img_files[START_INDEX:]
        print(f"📐 開炸 MediaPipe 流水線...")

        saved_counter = 0
        for file_path, cat_name in tqdm(sliced_img_files, desc="10000張平衡清洗與格式化中"):
            
            # 🛑 煞車機制
            if cat_name in TARGET_CATEGORIES and saved_counters[cat_name] >= MAX_PER_TARGET:
                continue
            if cat_name not in TARGET_CATEGORIES and saved_counters[cat_name] >= MAX_PER_NOISE:
                continue

            try:
                original_filename = file_path.split('/')[-1]
                base_name = os.path.splitext(original_filename)[0]

                # 💡 洗 Label 邏輯：判定它是 1~5 還是 0 (N_A)
                if cat_name in TARGET_CATEGORIES:
                    label_name = cat_name
                    numeric_label = LABEL_MAP[cat_name]
                else:
                    label_name = "N_A"
                    numeric_label = 0
                
                # 💡 防撞名設計：不管什麼類別，檔名最前面都加上原始資料夾名稱
                final_base_name = f"{cat_name}_{base_name}"

                # 唯讀串流過水 MediaPipe
                img_bytes = z.read(file_path)
                pil_img = Image.open(io.BytesIO(img_bytes))
                
                result = preprocessor.preprocess_image(pil_img)
                if result is None:
                    continue  # 沒抓到手直接放生
                    
                crop_img, landmarks = result
                # crop_resized = cv2.resize(crop_img, (224, 224))

                # 💡 實體寫入到新的扁平化資料夾
                img_out_path = crop_dir / f"{final_base_name}.jpg"
                npy_out_path = landmark_dir / f"{final_base_name}.npy"
                
                cv2.imwrite(str(img_out_path), cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
                np.save(str(npy_out_path), landmarks)

                # 💡 將這筆成功的資料記錄到 CSV 名單中
                records.append({
                    "idx": final_base_name,
                    "original_class_folder": cat_name,
                    "label": numeric_label,
                    "label_name": label_name,
                    "crop_path": str(img_out_path),
                    "landmark_path": str(npy_out_path),
                    "quality": "ok"
                })

                saved_counters[cat_name] += 1
                saved_counter += 1

                # 檢查是否所有類別都集滿了
                all_done = True
                for c in all_detected_cats:
                    limit = MAX_PER_TARGET if c in TARGET_CATEGORIES else MAX_PER_NOISE
                    if saved_counters[c] < limit:
                        all_done = False
                        break
                if all_done:
                    print("\n🎯 賀！本資料集已完美集滿黃金配額！提前收網！")
                    break

            except Exception:
                continue

    # ==================== 📊 最終產出 CSV 與報告 ====================
    print("\n📝 正在生成 PyTorch 專用標籤對照表 (labels.csv)...")
    df = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "labels.csv"
    df.to_csv(csv_path, index=False)

    total_target_saved = sum(saved_counters[c] for c in TARGET_CATEGORIES if c in saved_counters)
    total_na_saved = sum(saved_counters[c] for c in saved_counters if c not in TARGET_CATEGORIES)
    
    print("\n==================================================")
    print(f"🎉 【一條龍建構完畢】！")
    print(f"📈 產出摘要：")
    print(f"   - 5大目標手勢總計：{total_target_saved} 張照片對")
    print(f"   - 29類均勻 N/A 總計：{total_na_saved} 張照片對 (Class 0)")
    print(f"📁 圖片位置：{crop_dir}")
    print(f"📁 座標位置：{landmark_dir}")
    print(f"🧾 總目錄 CSV：{csv_path}")
    print("==================================================")

if __name__ == "__main__":
    main()