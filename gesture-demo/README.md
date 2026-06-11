# 手勢辨識 Demo — Team 16

即時手勢辨識網頁，使用 MediaPipe Hands + ONNX Runtime Web，完全在瀏覽器端運行，無需後端。

支援手勢：**fist · like · ok · one · palm**

---

## 部署步驟（10 分鐘搞定）

### Step 1：匯出模型

```bash
# 在你的 Python 環境執行
python export_onnx.py
```

會在 `model/` 資料夾產生 `landmark_only.onnx`（約 100–200 KB）。

### Step 2：建立 GitHub Repo

```
gesture-demo/
├── index.html          ← 主頁面（已完成）
├── export_onnx.py      ← 一次性轉換腳本
└── model/
    └── landmark_only.onnx   ← 你 export 出來的模型
```

```bash
git init
git add .
git commit -m "init gesture demo"
git branch -M main
git remote add origin https://github.com/你的帳號/gesture-demo.git
git push -u origin main
```

### Step 3：開啟 GitHub Pages

1. 進入 repo → **Settings** → **Pages**
2. Source 選 **Deploy from a branch**
3. Branch 選 **main**，資料夾選 **/ (root)**
4. 點 Save，等 1–2 分鐘

你的連結會是：`https://你的帳號.github.io/gesture-demo/`

分享這個連結給朋友就好了！

---

## 備注

- 沒有 `landmark_only.onnx` 時，網頁會自動降回幾何特徵 fallback（效果稍差但可用）。
- 只在 HTTPS 下才能存取攝影機（GitHub Pages 預設是 HTTPS，OK）。
- 模型跑在瀏覽器 WebAssembly，不需要 GPU。
