import glob, os, torch

SNAP_DIR = "data/processed/weekly_pt_scaled"
paths = sorted(glob.glob(os.path.join(SNAP_DIR, "*.pt")))

def load_pt(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(p, map_location="cpu")

d = load_pt(paths[0])
keys = sorted(list(d.keys()))
print("Keys:", keys)

# tìm meta scaler
scaler_keys = [k for k in keys if "scal" in k.lower() or "mean" in k.lower() or "std" in k.lower()]
print("Scaler-like keys:", scaler_keys)