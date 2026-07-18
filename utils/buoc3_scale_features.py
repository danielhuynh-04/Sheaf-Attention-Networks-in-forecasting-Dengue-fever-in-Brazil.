# utils/buoc3_scale_features.py
# ------------------------------
# BƯỚC 3: Scale features (train-only), log-transform label
# - Input  : data/processed/weekly_pt/*.pt  (tạo từ Bước 2)
# - Output : data/processed/weekly_pt_scaled/*.pt  (x đã scale, y=log1p(casos))
# - Meta   : data/processed/scaler_weekly.json  (tham số scaler theo feature)
#
# Ghi chú:
# - Không dùng 'date' (đã bỏ ở Bước 2). Chỉ dựa vào year+epiweek.
# - Scaler được ước lượng CHỈ từ năm train (2010-2020) để tránh leakage.
# - Hỗ trợ 'standard' (mặc định) hoặc 'minmax'.
# - Label: y' = log1p(y). Không scale thêm.
# - Giữ nguyên thứ tự node/feature_cols/masks; chỉ thay x và y.

import os
import json
import glob
import math
import torch
import numpy as np

PROCESSED_DIR = "data/processed"
SRC_DIR = os.path.join(PROCESSED_DIR, "weekly_pt")
DST_DIR = os.path.join(PROCESSED_DIR, "weekly_pt_scaled")
SCALER_JSON = os.path.join(PROCESSED_DIR, "scaler_weekly.json")

os.makedirs(DST_DIR, exist_ok=True)

# ------------------------------
# Cấu hình scaler
# ------------------------------
SCALER_TYPE = "standard"  # "standard" | "minmax"
EPS = 1e-6

def is_train_year(y: int) -> bool:
    return 2010 <= y <= 2020

def parse_year_week_from_fname(fname: str):
    # ví dụ: .../2017_14.pt hoặc 2017_14_something.pt
    base = os.path.basename(fname)
    name, _ = os.path.splitext(base)
    parts = name.split("_")
    if len(parts) < 2:
        return None, None
    try:
        yy = int(parts[0])
        ww = int(parts[1])
        return yy, ww
    except:
        return None, None

# ------------------------------
# 1) Quét file, đọc meta để lấy số feature & feature_cols
# ------------------------------
pt_files = sorted(glob.glob(os.path.join(SRC_DIR, "*.pt")))
if not pt_files:
    raise FileNotFoundError(f"Không tìm thấy snapshot .pt trong {SRC_DIR}. Hãy chạy Bước 2 trước.")

# Lấy một file mẫu để biết feature_cols
sample = torch.load(pt_files[0], map_location="cpu", weights_only=False)
feature_cols = sample.get("feature_cols", None)
if feature_cols is None:
    raise KeyError("Thiếu 'feature_cols' trong snapshot .pt (Bước 2).")

F = len(feature_cols)
print(f"📦 Bước 3 — Scale features ({SCALER_TYPE}), log1p(y). Số feature: {F}")

# ------------------------------
# 2) Ước lượng tham số scaler từ TRAIN (2010-2020)
#    - Standard: mean, std (per-feature)
#    - MinMax  : min, max (per-feature)
# ------------------------------
if SCALER_TYPE == "standard":
    # sum, sumsq, count để tính mean/std ổn định
    sum_feat = np.zeros((F,), dtype=np.float64)
    sumsq_feat = np.zeros((F,), dtype=np.float64)
    count = 0
elif SCALER_TYPE == "minmax":
    min_feat = np.full((F,), np.inf, dtype=np.float64)
    max_feat = np.full((F,), -np.inf, dtype=np.float64)
else:
    raise ValueError("SCALER_TYPE phải là 'standard' hoặc 'minmax'.")

train_files = []
for p in pt_files:
    yy, ww = parse_year_week_from_fname(p)
    if yy is None:
        continue
    if is_train_year(yy):
        train_files.append(p)

if not train_files:
    raise RuntimeError("Không tìm thấy file TRAIN (2010-2020) để ước lượng scaler.")

for p in train_files:
    d = torch.load(p, map_location="cpu", weights_only=False)
    x = d["x"].cpu().numpy().astype(np.float64)  # (N,F)
    # bảo đảm shape hợp lệ
    if x.ndim != 2 or x.shape[1] != F:
        raise ValueError(f"x shape không khớp số feature ở {p}: {x.shape} vs F={F}")

    if SCALER_TYPE == "standard":
        sum_feat += x.sum(axis=0)
        sumsq_feat += (x * x).sum(axis=0)
        count += x.shape[0]
    elif SCALER_TYPE == "minmax":
        min_feat = np.minimum(min_feat, np.nanmin(x, axis=0))
        max_feat = np.maximum(max_feat, np.nanmax(x, axis=0))

if SCALER_TYPE == "standard":
    mean = sum_feat / max(count, 1)
    var = (sumsq_feat / max(count, 1)) - (mean * mean)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    std = np.where(std < EPS, 1.0, std)  # tránh chia 0
    scaler_params = {"type": "standard",
                     "mean": mean.tolist(),
                     "std": std.tolist(),
                     "feature_cols": feature_cols,
                     "train_count_rows": int(count)}
elif SCALER_TYPE == "minmax":
    # tránh TH min == max
    span = max_feat - min_feat
    span = np.where(span < EPS, 1.0, span)
    scaler_params = {"type": "minmax",
                     "min": min_feat.tolist(),
                     "max": max_feat.tolist(),
                     "feature_cols": feature_cols}

with open(SCALER_JSON, "w", encoding="utf-8") as f:
    json.dump(scaler_params, f, ensure_ascii=False, indent=2)
print(f"✅ Lưu tham số scaler: {SCALER_JSON}")

# ------------------------------
# 3) Áp dụng scaler cho MỌI snapshot và log1p cho y
#    - Ghi ra thư mục weekly_pt_scaled
# ------------------------------
n_written = 0

for p in pt_files:
    d = torch.load(p, map_location="cpu", weights_only=False)

    x = d["x"].cpu().numpy().astype(np.float32)  # (N,F)
    y = d["y"].cpu().numpy().astype(np.float32)  # (N,)

    # scale X
    if SCALER_TYPE == "standard":
        mean = np.array(scaler_params["mean"], dtype=np.float32)
        std  = np.array(scaler_params["std"], dtype=np.float32)
        x_scaled = (x - mean) / std
    else:  # minmax
        _min = np.array(scaler_params["min"], dtype=np.float32)
        _max = np.array(scaler_params["max"], dtype=np.float32)
        span = _max - _min
        span = np.where(span < EPS, 1.0, span).astype(np.float32)
        x_scaled = (x - _min) / span  # in [0,1]

    # log1p cho y
    y_log = np.log1p(np.maximum(y, 0.0, dtype=np.float32))

    # tạo bản sao dict để ghi
    out = dict(d)
    out["x"] = torch.tensor(x_scaled, dtype=torch.float32)
    out["y"] = torch.tensor(y_log, dtype=torch.float32)
    out["label_transform"] = "log1p"
    out["scaler"] = scaler_params

    fname = os.path.basename(p)
    torch.save(out, os.path.join(DST_DIR, fname))
    n_written += 1

print(f"✅ Đã scale & lưu {n_written} snapshot @ {DST_DIR}")

# ------------------------------
# 4) Kiểm tra nhanh 1 file mẫu
# ------------------------------
sample_out = os.path.join(DST_DIR, os.path.basename(pt_files[0]))
chk = torch.load(sample_out, map_location="cpu", weights_only=False)
xs = chk["x"].numpy()
ys = chk["y"].numpy()

# sanity
nan_x = np.isnan(xs).sum()
inf_x = np.isinf(xs).sum()
nan_y = np.isnan(ys).sum()
inf_y = np.isinf(ys).sum()

print("Sanity (sample):")
print(f"  x shape={xs.shape}, NaN={nan_x}, Inf={inf_x}")
print(f"  y shape={ys.shape}, NaN={nan_y}, Inf={inf_y}")
print(f"  feature_cols={len(chk.get('feature_cols', []))}, label_transform={chk.get('label_transform')}")
print("🎉 Bước 3 hoàn tất.")
