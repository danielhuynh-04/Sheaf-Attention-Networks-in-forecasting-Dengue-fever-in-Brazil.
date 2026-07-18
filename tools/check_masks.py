import glob, os, torch

SNAP_DIR = "data/processed/weekly_pt_scaled"
paths = sorted(glob.glob(os.path.join(SNAP_DIR, "*.pt")))

def load_pt(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(p, map_location="cpu")

# check 20 file rải đều
pick = paths[::max(1, len(paths)//20)]

for p in pick:
    d = load_pt(p)
    tr = d.get("train_mask")
    va = d.get("val_mask")
    te = d.get("test_mask")

    if tr is None or va is None or te is None:
        print("Missing mask in:", os.path.basename(p))
        continue

    ov_tv = (tr & va).sum().item()
    ov_tt = (tr & te).sum().item()
    ov_vt = (va & te).sum().item()

    if (ov_tv + ov_tt + ov_vt) > 0:
        print("❌ Overlap:", os.path.basename(p), ov_tv, ov_tt, ov_vt)

print("✅ Done (no output above = no overlap).")