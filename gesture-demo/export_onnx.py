"""
export_onnx.py
把 landmark_only.pth 轉成 landmark_only.onnx
在你的 Python 環境執行一次即可：python export_onnx.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Windows + conda OpenMP 衝突 workaround

import torch
from pathlib import Path

# ── 貼上你的 model 定義 ────────────────────────────────────────────────
import torch.nn as nn

class LandmarkOnlyModel(nn.Module):
    def __init__(self, input_dim=42, num_classes=6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        return self.mlp(x)
# ─────────────────────────────────────────────────────────────────────────

PTH  = Path("model/landmark_only.pth")
ONNX = Path("model/landmark_only.onnx")

print(f"載入 {PTH} ...")
model = LandmarkOnlyModel(input_dim=42, num_classes=6)
state = torch.load(PTH, map_location="cpu")
state = {k: v.float() if torch.is_floating_point(v) else v for k, v in state.items()}
model.load_state_dict(state)
model.eval()

dummy = torch.zeros(1, 42)

print(f"匯出 → {ONNX} ...")
torch.onnx.export(
    model, dummy, str(ONNX),
    input_names=["landmarks"],
    output_names=["logits"],
    dynamic_axes={"landmarks": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=11,
)

size_kb = ONNX.stat().st_size / 1024
print(f"✅ 完成！檔案大小：{size_kb:.1f} KB")
print(f"   把 {ONNX} 放到 GitHub repo 的 model/ 資料夾即可。")