import glob, os, torch

SNAP_DIR = "data/processed/weekly_pt_scaled"
paths = sorted(glob.glob(os.path.join(SNAP_DIR, "*.pt")))

# weights_only=True (nếu snapshot chỉ chứa tensor/list đơn giản thì OK)
try:
    d = torch.load(paths[0], map_location="cpu", weights_only=True)
except TypeError:
    d = torch.load(paths[0], map_location="cpu")

cols = d.get("feature_cols", [])
print("Num feature cols:", len(cols))

bad_keywords = ["lead", "future", "next", "t+","lag_-"]
bad = [c for c in cols if any(k in str(c).lower() for k in bad_keywords)]
print("Suspicious cols:", bad[:50])