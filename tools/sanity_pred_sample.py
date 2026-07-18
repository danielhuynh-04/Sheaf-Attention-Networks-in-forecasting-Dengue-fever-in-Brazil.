import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch, glob, os
import numpy as np

SNAP_DIR = "data/processed/weekly_pt_scaled"
EDGE_PATH = "data/processed/edge_index.pt"

from trainers.trainer_weekly import build_temporal_seq
from models.model_factory import build_model

DEVICE="cpu"

def load_pt(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(p, map_location="cpu")

# lấy 1 snapshot val
paths = sorted(glob.glob(os.path.join(SNAP_DIR, "*.pt")))
valp = [p for p in paths if os.path.basename(p).startswith(("2021","2022"))][0]
d = load_pt(valp)

x = d["x"].to(DEVICE)
y = d["y"].to(DEVICE).view(-1)
mask = d["val_mask"].to(DEVICE)

edge = load_pt(EDGE_PATH).to(DEVICE)

model = build_model("sheaf_conn", in_dim=x.shape[1], hidden=64, out_dim=1, dropout=0.2).to(DEVICE)
ckpt = load_pt("checkpoints/sheaf_conn_global_best.pt")
model.load_state_dict(ckpt["state_dict"])
model.eval()

feature_cols = d.get("feature_cols", [])
temporal_seq, _ = build_temporal_seq(x, feature_cols)

with torch.no_grad():
    yhat = model(x, edge, temporal_seq=temporal_seq).view(-1)

idx = mask.nonzero(as_tuple=False).view(-1)[:5]
print("Sample file:", os.path.basename(valp))
for i in idx.tolist():
    print(i, "y=", float(y[i]), "yhat=", float(yhat[i]), "abs_err=", float(abs(yhat[i]-y[i])))