import os
import glob
import json
import argparse
import random
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- add ROOT to sys.path để import nội bộ không lỗi
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from trainers.trainer_weekly import build_temporal_seq, make_model, _find_lag_columns
from models.model_factory import build_model
from evaluation.metrics import evaluate_regression, trimmed_r2


# ---------------------- config ----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SNAP_DIR = "data/processed/weekly_pt_scaled"
EDGE_PATH = "data/processed/edge_index.pt"

OUT_DIR = "data/interim/permutation"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------- utils ----------------------
def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def year_from_file(path: str) -> int:
    return int(os.path.basename(path).split("_")[0])


def list_snapshots() -> List[str]:
    paths = sorted(glob.glob(os.path.join(SNAP_DIR, "*.pt")))
    if not paths:
        raise FileNotFoundError(f"Không thấy *.pt trong {SNAP_DIR}")
    return paths


def split_weeks(paths: List[str]) -> Tuple[List[str], List[str], List[str]]:
    tr, va, te = [], [], []
    for p in paths:
        y = year_from_file(p)
        if 2010 <= y <= 2020:
            tr.append(p)
        elif y in (2021, 2022):
            va.append(p)
        elif y in (2023, 2024):
            te.append(p)
    return tr, va, te


def load_edge() -> torch.Tensor:
    e = safe_load(EDGE_PATH)
    return e.to(DEVICE)


def squeeze_1d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 2 and t.size(-1) == 1:
        return t.squeeze(-1)
    return t


# ---------------------- model builder (match your project) ----------------------
def build_selected_model(model_name: str, in_dim: int, has_temporal: bool):
    """
    Must match how you trained.
    - gat uses make_model from trainer_weekly
    - others use build_model factory
    """
    if model_name == "gat":
        return make_model(
            in_dim=in_dim,
            has_temporal=has_temporal,
            hidden=128,
            heads=(4, 4),
            gat_dropout=0.2
        ).to(DEVICE)
    else:
        # IMPORTANT: match hidden for sheaf_conn (you trained hidden=64)
        if model_name == "sheaf_conn":
            hidden = 64
            lr = 1e-4
            wd = 5e-4
        else:
            hidden = 128
            lr = 3e-4
            wd = 1e-4

        model = build_model(
            model_name,
            in_dim=in_dim,
            hidden=hidden,
            out_dim=1,
            dropout=0.2
        ).to(DEVICE)
        return model


def load_best_checkpoint(model_name: str):
    ckpt_path = os.path.join("checkpoints", f"{model_name}_global_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Không thấy checkpoint: {ckpt_path}")
    ckpt = safe_load(ckpt_path)
    return ckpt, ckpt_path


# ---------------------- evaluation core ----------------------
@torch.no_grad()
def eval_split(model, edge_index, paths: List[str], split: str) -> Dict[str, float]:
    """
    Evaluate in log-space on given split masks (val/test).
    Return micro metrics (over all nodes+weeks).
    """
    y_true_all = []
    y_pred_all = []

    model.eval()

    for p in paths:
        d = safe_load(p)
        x = d["x"].to(DEVICE)
        y = d["y"].to(DEVICE)

        mask = d.get(f"{split}_mask", None)
        if mask is None:
            raise KeyError(f"Thiếu {split}_mask trong {p}")
        mask = mask.to(DEVICE)
        if mask.sum().item() == 0:
            continue

        feature_cols = d.get("feature_cols", [])
        temporal_seq, _ = build_temporal_seq(x, feature_cols)

        yhat = model(x, edge_index, temporal_seq=temporal_seq)

        y = squeeze_1d(y)
        yhat = squeeze_1d(yhat)

        y_true_all.append(y[mask].detach().cpu().view(-1))
        y_pred_all.append(yhat[mask].detach().cpu().view(-1))

    yt = torch.cat(y_true_all, 0)
    yp = torch.cat(y_pred_all, 0)

    m = evaluate_regression(yp, yt)
    m["R2trim"] = float(trimmed_r2(yt.numpy(), yp.numpy(), trim=0.01))
    return {
        "MAE_log": float(m["MAE"]),
        "RMSE_log": float(m["RMSE"]),
        "R2_log": float(m["R2"]),
        "R2trim_log": float(m["R2trim"]),
        "n": int(yt.numel())
    }


def permute_train_labels_in_memory(d: dict, rng: np.random.Generator) -> dict:
    """
    Return a shallow-copied dict with y permuted ONLY on train_mask.
    Does NOT modify original file on disk.
    """
    out = dict(d)
    y = out["y"].clone()
    y = squeeze_1d(y)

    m = out.get("train_mask", None)
    if m is None:
        raise KeyError("Thiếu train_mask trong snapshot")

    idx = torch.where(m)[0].cpu().numpy()
    if idx.size > 1:
        y_np = y[idx].cpu().numpy()
        rng.shuffle(y_np)
        y[idx] = torch.tensor(y_np, dtype=y.dtype)

    out["y"] = y
    return out


def quick_fit_on_permuted_train(model_name: str, model, edge_index, train_paths: List[str],
                                epochs: int, seed: int) -> None:
    """
    Train a few epochs on permuted TRAIN labels (in-memory) to see if performance on val/test collapses.
    This does NOT alter your saved checkpoints or files.
    """
    rng = np.random.default_rng(seed)
    model.train(True)

    # optimizer: keep same style as your run_global_gat
    if model_name == "sheaf_conn":
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    loss_fn = nn.HuberLoss(delta=1.2)

    # sample subset for speed (optional): comment out if you want full train
    # train_paths = train_paths  # full
    # For fast permutation test, you can subsample weeks:
    # train_paths = train_paths[::2]

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        steps = 0

        for p in train_paths:
            d0 = safe_load(p)
            d = permute_train_labels_in_memory(d0, rng)

            x = d["x"].to(DEVICE)
            y = d["y"].to(DEVICE)
            m = d["train_mask"].to(DEVICE)
            if m.sum().item() == 0:
                continue

            feature_cols = d.get("feature_cols", [])
            temporal_seq, _ = build_temporal_seq(x, feature_cols)

            yhat = model(x, edge_index, temporal_seq=temporal_seq)
            y = squeeze_1d(y)
            yhat = squeeze_1d(yhat)

            loss = loss_fn(yhat[m].view(-1), y[m].view(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.item())
            steps += 1

        if steps > 0 and (ep == 1 or ep == epochs):
            print(f"[perm-train] Epoch {ep}/{epochs} | loss={total_loss/steps:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="sheaf_conn",
                    choices=["gnn", "gcn", "gat", "sheaf", "sheaf_conn"])
    ap.add_argument("--perm_epochs", type=int, default=10,
                    help="Số epoch train trên nhãn permuted (nhanh thôi).")
    ap.add_argument("--n_perm", type=int, default=5,
                    help="Số lần permutation (nhiều hơn thì chắc hơn nhưng lâu hơn).")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    seed_all(args.seed)

    # data
    paths = list_snapshots()
    tr_paths, va_paths, te_paths = split_weeks(paths)

    # infer in_dim/temporal
    sample = safe_load(tr_paths[0])
    in_dim = int(sample["x"].shape[1])
    has_temporal = bool(_find_lag_columns(sample.get("feature_cols", [])))

    edge_index = load_edge()

    # build model + load best checkpoint (baseline)
    model = build_selected_model(args.model, in_dim=in_dim, has_temporal=has_temporal)
    ckpt, ckpt_path = load_best_checkpoint(args.model)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)

    print("Model:", args.model, "| ckpt:", ckpt_path)
    print("Train weeks:", len(tr_paths), "Val weeks:", len(va_paths), "Test weeks:", len(te_paths))

    # baseline eval
    base_val = eval_split(model, edge_index, va_paths, "val")
    base_test = eval_split(model, edge_index, te_paths, "test")
    print("\n[BASELINE] val:", base_val)
    print("[BASELINE] test:", base_test)

    # permutation runs
    rows = []
    for k in range(args.n_perm):
        perm_seed = args.seed + 1000 + k
        print(f"\n=== Permutation run {k+1}/{args.n_perm} | seed={perm_seed} ===")

        # fresh model from same architecture + load same ckpt to start
        m2 = build_selected_model(args.model, in_dim=in_dim, has_temporal=has_temporal)
        m2.load_state_dict(ckpt["state_dict"])
        m2.to(DEVICE)

        # train a few epochs on permuted labels
        quick_fit_on_permuted_train(args.model, m2, edge_index, tr_paths, epochs=args.perm_epochs, seed=perm_seed)

        # evaluate true val/test labels
        pv = eval_split(m2, edge_index, va_paths, "val")
        pt = eval_split(m2, edge_index, te_paths, "test")
        print("[PERM] val:", pv)
        print("[PERM] test:", pt)

        rows.append({
            "model": args.model,
            "perm_run": k + 1,
            "perm_seed": perm_seed,
            "perm_epochs": args.perm_epochs,

            "base_val_R2_log": base_val["R2_log"],
            "base_test_R2_log": base_test["R2_log"],

            "perm_val_R2_log": pv["R2_log"],
            "perm_test_R2_log": pt["R2_log"],

            "perm_val_MAE_log": pv["MAE_log"],
            "perm_test_MAE_log": pt["MAE_log"],
        })

    # save results
    out_csv = os.path.join(OUT_DIR, f"permutation_{args.model}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")

    out_json = os.path.join(OUT_DIR, f"permutation_{args.model}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "baseline": {"val": base_val, "test": base_test},
            "runs": rows
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved: {out_csv}")
    print(f"✅ Saved: {out_json}")
    print("\nInterpretation:")
    print("- If permuted R2 drops near 0/negative and MAE increases => no obvious shortcut/leak.")
    print("- If permuted R2 stays close to baseline => investigate leakage/bug.")


if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("Please install pandas: pip install pandas")
    main()