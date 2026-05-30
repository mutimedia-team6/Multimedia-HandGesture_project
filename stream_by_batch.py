import zipfile
import io
import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# 正確引入助教給的 hand_preprocess.py 裡面的類別
from hand_preprocess import MediaPipeHandPreprocessor 

# ==================== 🛠️ 分批控制面板 ====================
ZIP_FILE_PATH = Path(r"/mnt/d/hagrid_project/hagridv2_512.zip")      # 119GB 原始大壓縮檔
OUTPUT_DIR = Path(r"/mnt/d/hagrid_project/data_all") # 最終大資料集目錄

# 💡 【第一波先洗專題最核心的 5 個目標手勢】 
BATCH_CATEGORIES = []

# 📝 備忘錄（HaGRID v2 完整的 34 個類別名稱，方便你之後複製貼上）：
# ['fist', 'like', 'ok', 'one', 'palm', 'call', 'dislike', 'fist', 'four', 'like', 
#  'mute', 'ok', 'one', 'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted', 
#  'three', 'three2', 'two_up', 'two_up_inverted', 'no_gesture', 'heavy', 'guitar', 'point_up', 
#  'point_down', 'point_left', 'point_right', 'fist_moved', 'palm_moved', 'hand_heart', 
#  'hand_heart_small', 'thumb_down', 'thumb_up', 'press', 'squeeze', 'swipe']
# ========================================================

def main():
    if not ZIP_FILE_PATH.exists():
        print(f"❌ 找不到原始壓縮檔，請檢查路徑：{ZIP_FILE_PATH}")
        return

    # 實例化助教的類別
    preprocessor = MediaPipeHandPreprocessor()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🚀 [分批戰術啟動] 目前指定清洗類別：{BATCH_CATEGORIES}")
    
    with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as z:
        print("🔍 正在掃描壓縮檔內符合的圖片...")
        all_files = z.namelist()
        
        # 精準篩選與目錄探測補丁
        img_files = []
        for f in all_files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and ".ipynb_checkpoints" not in f:
                parts = f.split('/')
                
                matched_cat = None
                for part in parts:
                    if part in BATCH_CATEGORIES:
                        matched_cat = part
                        break
                
                if matched_cat:
                    # 儲存成 tuple 格式
                    img_files.append((f, matched_cat))
                    
        if not img_files:
            print("⚠️ 沒有找到符合目前設定類別的圖片，請檢查資料夾名稱是否拼錯！")
            return
            
        print(f"📸 本次總計有 {len(img_files)} 張原始圖片準備進行記憶體清洗。")

        # 3. 串流前處理（完美格式解構對齊 👑）
        saved_counter = 0
        
        # 修正核心：在這裡用 file_path, cat_name 完美把 tuple 解開！
        for file_path, cat_name in tqdm(img_files, desc="當前批次處理中"):
            try:
                # 檔名解碼，保持原廠
                original_filename = file_path.split('/')[-1] 
                base_name = os.path.splitext(original_filename)[0]
                
                # 自動建立對應手勢的資料夾
                cat_output_dir = OUTPUT_DIR / cat_name
                cat_output_dir.mkdir(parents=True, exist_ok=True)
                
                # 串流讀取（傳入真正的字串路徑，不再是 tuple 物件！）
                img_bytes = z.read(file_path)
                pil_img = Image.open(io.BytesIO(img_bytes))
                
                # 呼叫助教的前處理函數
                result = preprocessor.preprocess_image(pil_img)
                
                if result is None:
                    continue
                    
                crop_img, landmarks = result
                
                # 降維打擊：縮放成 224x224
                crop_resized = cv2.resize(crop_img, (224, 224))
                
                # 設定輸出路徑（完美與原廠格式對齊）
                img_out_path = cat_output_dir / f"{base_name}.jpg"
                npy_out_path = cat_output_dir / f"{base_name}.npy"
                
                # 寫入硬碟
                cv2.imwrite(str(img_out_path), cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR))
                np.save(str(npy_out_path), landmarks)
                
                saved_counter += 1
                
            except Exception as e:
                # 如果你想看有沒有拋出什麼隱藏錯誤，可以把下面這一行的註解 # 拿掉
                # print(f"Error processing {file_path}: {e}")
                continue

    print(f"\n🎉 【本批次處理完成】！成功產出 {saved_counter} 組精華數據。")
    print(f"📁 已安全儲存至：{OUTPUT_DIR}")
    print(f"💡 隨時可以關閉視窗或移動電腦。下次要跑其他類別時，更換 BATCH_CATEGORIES 即可！")

if __name__ == "__main__":
    main()