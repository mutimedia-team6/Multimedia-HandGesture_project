import zipfile
import io
import os
import random
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# 正確引入助教給的 hand_preprocess.py 裡面的類別
from hand_preprocess import MediaPipeHandPreprocessor 

# ==================== 🛠️ 10000張滑塊控制面板 ====================
ZIP_FILE_PATH = Path(r"/mnt/d/hagrid_project/hagridv2_512.zip")      

# 💡 這次我們輸出成中型版本 v1_medium
OUTPUT_DIR = Path(r"/mnt/d/hagrid_project/hagrid_set_v3")     
START_INDEX = 0                                                    

# 5 個專題指定的標準目標手勢
TARGET_CATEGORIES = {'fist', 'like', 'ok', 'one', 'palm'}

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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 建立目標類別資料夾與 0_NA 資料夾
    for cat in TARGET_CATEGORIES:
        (OUTPUT_DIR / cat).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "0_NA").mkdir(parents=True, exist_ok=True)

    print(f"🚀 [滑塊窗口大數據流水線] 目錄：{OUTPUT_DIR} | 起始點控制：{START_INDEX}")
    
    saved_counters = {}

    with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as z:
        print("🔍 步驟一：正在建立壓縮檔全量索引...")
        all_files = z.namelist()
        
        # 👑 引入你那支成功腳本的「多層路徑智慧探測」
        raw_img_files = []
        for f in all_files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and ".ipynb_checkpoints" not in f:
                parts = f.split('/')
                
                # 逆向從路徑倒數第二層開始找手勢名字（通常是資料夾名）
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
        
        # 🔥 【固定 Seed 全局場景大打亂】打破人臉、房間聚集的魔咒
        print(f"🎴 正在注入 Seed {RANDOM_SEED} 進行全局打亂（提升數據多樣性）...")
        random.seed(RANDOM_SEED)
        random.shuffle(raw_img_files)
        
        # 👑 【發動滑塊切片】
        sliced_img_files = raw_img_files[START_INDEX:]
        print(f"📐 已跳過前 {START_INDEX} 張照片，剩餘 {len(sliced_img_files)} 張候選圖，開炸 MediaPipe 流水線...")

        # 步驟三：串流前處理與煞車
        saved_counter = 0
        for file_path, cat_name in tqdm(sliced_img_files, desc="10000張平衡清洗中"):
            
            # 🛑 煞車機制
            if cat_name in TARGET_CATEGORIES and saved_counters[cat_name] >= MAX_PER_TARGET:
                continue
            if cat_name not in TARGET_CATEGORIES and saved_counters[cat_name] >= MAX_PER_NOISE:
                continue

            try:
                original_filename = file_path.split('/')[-1]
                base_name = os.path.splitext(original_filename)[0]

                # 洗 Label 邏輯：目標類別維持原名，29類雜訊全包進 0_NA
                if cat_name in TARGET_CATEGORIES:
                    cat_output_dir = OUTPUT_DIR / cat_name
                    final_base_name = base_name
                else:
                    cat_output_dir = OUTPUT_DIR / "0_NA"
                    final_base_name = f"{cat_name}_{base_name}" # 前綴防撞名

                # 唯讀串流過水 MediaPipe
                img_bytes = z.read(file_path)
                pil_img = Image.open(io.BytesIO(img_bytes))
                
                result = preprocessor.preprocess_image(pil_img)
                if result is None:
                    continue  # 沒抓到手直接放生
                    
                crop_img, landmarks = result
                crop_resized = cv2.resize(crop_img, (224, 224))

                # 實體寫入
                img_out_path = cat_output_dir / f"{final_base_name}.jpg"
                npy_out_path = cat_output_dir / f"{final_base_name}.npy"
                
                cv2.imwrite(str(img_out_path), cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR))
                np.save(str(npy_out_path), landmarks)

                saved_counters[cat_name] += 1
                saved_counter += 1

                # 檢查是否所有類別（包括 5 個目標與 29 個雜訊）都各自集滿了
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

    # ==================== 📊 最終成果報告 ====================
    total_target_saved = sum(saved_counters[c] for c in TARGET_CATEGORIES if c in saved_counters)
    total_na_saved = sum(saved_counters[c] for c in saved_counters if c not in TARGET_CATEGORIES)
    
    print("\n==================================================")
    print(f"🎉 【10,000張黃金平衡資料集建構完畢】！")
    print(f"📈 本出版版（{OUTPUT_DIR.name}）統計摘要：")
    print(f"   - 5大目標手勢總計：{total_target_saved} 張照片對")
    print(f"   - 29類均勻 N/A 總計：{total_na_saved} 張照片對 (Class 0 防禦力拉滿)")
    print(f"📁 本組儲存路徑：{OUTPUT_DIR}")
    print("==================================================")

if __name__ == "__main__":
    main()