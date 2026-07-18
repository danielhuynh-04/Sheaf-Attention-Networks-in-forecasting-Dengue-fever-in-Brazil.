# run_global_gat.py
# ------------------------------------------------------------
# Global Graph Model theo tuần (CPU/GPU), HuberLoss, early stopping,
# metric đầy đủ (log/real, R2 trim), AUC/PR từ hồi quy.
# Bias-correction: Duan smearing (ưu tiên) + fallback sigma^2.
# Thêm clamp theo phân vị cao của TRAIN để giảm outlier ở miền thực.
#
# OUTPUT chính cho paper:
# - data/interim/<model>_global_weekly_report.csv
# - data/interim/<model>_global_summary.json
# - data/interim/<model>_epoch_log.csv
# - checkpoints/<model>_global_best.pt
#
# OUTPUT trực quan hoá (tuỳ chọn, KHÔNG PHÌNH Ổ CỨNG):
# - visualizations/data/<model>/node_predictions_<model>.csv
#   (CHỈ export 1 lần ở cuối nếu --export_predictions 1)
# ------------------------------------------------------------
from __future__ import annotations
import os
import glob
import time
import json
import random
import argparse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.trainer_weekly import build_temporal_seq, make_model, _find_lag_columns
from models.model_factory import build_model
from evaluation.metrics import (
    evaluate_regression,
    trimmed_r2,
    classification_metrics_from_regression
)

# ---------------------- cấu hình ----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SNAP_DIR = "data/processed/weekly_pt_scaled"
EDGE_PATH = "data/processed/edge_index.pt"

REPORT_DIR = "data/interim"
CKPT_DIR = "checkpoints"
VIS_DIR = "visualizations"
VIS_DATA_DIR = os.path.join(VIS_DIR, "data")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(VIS_DATA_DIR, exist_ok=True)


# ---------------------- utils ----------------------
def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_edge(path: str) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def year_from_file(path: str) -> int:
    base = os.path.basename(path)
    return int(base.split("_")[0])


def list_snapshots() -> List[str]:
    paths = sorted(glob.glob(os.path.join(SNAP_DIR, "*.pt")))
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy *.pt trong {SNAP_DIR}")
    return paths


def split_weeks(paths: List[str]) -> Tuple[List[str], List[str], List[str]]:
    train, val, test = [], [], []
    for p in paths:
        y = year_from_file(p)
        if 2010 <= y <= 2020:
            train.append(p)
        elif y in (2021, 2022):
            val.append(p)
        elif y in (2023, 2024):
            test.append(p)
    return train, val, test


def append_epoch_log(path_csv: str, row: dict):
    df = pd.DataFrame([row])
    header = not os.path.exists(path_csv)
    df.to_csv(path_csv, mode="a", header=header, index=False, encoding="utf-8")


# ---------------------- backtransform helpers ----------------------
@torch.no_grad()
def backtransform_smear(y_log: torch.Tensor, smear: float | None) -> torch.Tensor:
    if smear is None or smear <= 0:
        return torch.expm1(y_log).clamp_min(0.0)
    return (torch.exp(y_log) * float(smear) - 1.0).clamp_min(0.0)


@torch.no_grad()
def backtransform_sigma(y_log: torch.Tensor, sigma2: float = 0.0) -> torch.Tensor:
    shift = 0.5 * float(max(0.0, sigma2))
    return torch.expm1(y_log + shift).clamp_min(0.0)


@torch.no_grad()
def apply_headroom_clamp(y_real: np.ndarray, cap: float | None) -> np.ndarray:
    if cap is None or cap <= 0:
        return y_real
    return np.clip(y_real, 0.0, float(cap))


# ---------------------- metrics block ----------------------
def _metrics_block(
    y_true_log_t: torch.Tensor,
    y_pred_log_t: torch.Tensor,
    smear: float | None,
    sigma2: float | None,
    cap: float | None
) -> Dict[str, float]:

    mlog = evaluate_regression(y_pred_log_t, y_true_log_t)
    r2t_log = trimmed_r2(y_true_log_t.numpy(), y_pred_log_t.numpy(), trim=0.01)

    yt_r = torch.expm1(y_true_log_t).clamp_min(0).numpy()
    if smear is not None and smear > 0:
        yp_r = backtransform_smear(y_pred_log_t, smear).cpu().numpy()
    else:
        yp_r = backtransform_sigma(y_pred_log_t, sigma2 or 0.0).cpu().numpy()

    yp_r = apply_headroom_clamp(yp_r, cap)

    mreal = evaluate_regression(yp_r, yt_r)
    r2t_real = trimmed_r2(yt_r, yp_r, trim=0.01)

    return {
        "MAE_log": mlog["MAE"],
        "RMSE_log": mlog["RMSE"],
        "R2_log": mlog["R2"],
        "R2trim_log": r2t_log,
        "MAE_real": mreal["MAE"],
        "RMSE_real": mreal["RMSE"],
        "SMAPE_real": mreal["SMAPE"],
        "R2_real": mreal["R2"],
        "R2trim_real": r2t_real,
    }


# ---------------------- epoch loop ----------------------
def run_epoch(
    model,
    edge_index: torch.Tensor,
    paths: List[str],
    phase: str = "train",
    opt: torch.optim.Optimizer | None = None,
    loss_fn: nn.Module | None = None,
    smear: float | None = None,
    sigma2: float | None = None,
    cap: float | None = None,
    write_predictions: bool = False,
    predictions_csv: str | None = None
) -> tuple[float, list[dict], dict | None]:

    is_train = (phase == "train")
    model.train(is_train)

    total_loss, n_steps = 0.0, 0
    weekly_rows: list[dict] = []
    mic_true_log, mic_pred_log = [], []

    for p in paths:
        d = _safe_load(p)
        x = d["x"].to(DEVICE)
        y = d["y"].to(DEVICE)

        mask = d.get(f"{phase}_mask", d.get(f"mask_{phase}", None))
        if mask is None:
            raise KeyError(f"Thiếu {phase}_mask trong {p}")
        mask = mask.to(DEVICE)
        if mask.sum().item() == 0:
            continue

        feature_cols = d.get("feature_cols", [])
        temporal_seq, _ = build_temporal_seq(x, feature_cols)

        with torch.set_grad_enabled(is_train):
            y_hat = model(x, edge_index, temporal_seq=temporal_seq)

            # ép shape về vector [N]
            if y_hat.dim() == 2 and y_hat.size(-1) == 1:
                y_hat = y_hat.squeeze(-1)
            if y.dim() == 2 and y.size(-1) == 1:
                y = y.squeeze(-1)

            y_pred = y_hat[mask].view(-1)
            y_true = y[mask].view(-1)

            loss_t = (
                loss_fn(y_pred, y_true)
                if loss_fn is not None
                else F.mse_loss(y_pred, y_true)
            )

            if is_train:
                opt.zero_grad()
                loss_t.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        total_loss += float(loss_t.item())
        n_steps += 1

        if phase in ("val", "test"):
            base = os.path.basename(p).replace(".pt", "")
            yr_s, ew_s = base.split("_")

            wk_m = _metrics_block(
                y[mask].view(-1).detach().cpu(),
                y_hat[mask].view(-1).detach().cpu(),
                smear=smear,
                sigma2=sigma2,
                cap=cap
            )

            weekly_rows.append({
                "Year": int(yr_s),
                "Epiweek": int(ew_s),
                "Split": phase,
                **wk_m
            })

            mic_true_log.append(y[mask].view(-1).detach().cpu())
            mic_pred_log.append(y_hat[mask].view(-1).detach().cpu())

        # CHỈ export node predictions ở eval cuối (write_predictions=True)
        if write_predictions and (predictions_csv is not None) and (phase in ("val", "test")) and ("geocodes" in d):
            with torch.no_grad():
                base = os.path.basename(p).replace(".pt", "")
                yr_s, ew_s = base.split("_")

                # dùng luôn y_hat, không forward lần 2
                yt_l = y.detach().cpu().view(-1)
                yp_l = y_hat.detach().cpu().view(-1)

                yt_r = torch.expm1(yt_l).clamp_min(0).numpy()
                if smear is not None and smear > 0:
                    yp_r = backtransform_smear(yp_l, smear).numpy()
                else:
                    yp_r = backtransform_sigma(yp_l, sigma2 or 0.0).numpy()
                yp_r = apply_headroom_clamp(yp_r, cap)

                idx = mask.cpu().numpy().astype(bool)
                geos = np.asarray(d["geocodes"])[idx]

                out_df = pd.DataFrame({
                    "Year": int(yr_s),
                    "Epiweek": int(ew_s),
                    "Split": phase,
                    "geocode": geos,
                    "y_true_log": yt_l.numpy()[idx],
                    "y_pred_log": yp_l.numpy()[idx],
                    "y_true": yt_r[idx],
                    "y_pred": yp_r[idx],
                })

                header = not os.path.exists(predictions_csv)
                out_df.to_csv(predictions_csv, mode="a", header=header, index=False, encoding="utf-8")

    avg_loss = total_loss / max(1, n_steps)

    micro = None
    if mic_true_log:
        yt = torch.cat(mic_true_log, 0)
        yp = torch.cat(mic_pred_log, 0)

        mic_log = evaluate_regression(yp, yt)
        mic_r2t_log = trimmed_r2(yt.numpy(), yp.numpy(), trim=0.01)

        yt_r = torch.expm1(yt).clamp_min(0).numpy()
        if smear is not None and smear > 0:
            yp_r = backtransform_smear(yp, smear).numpy()
        else:
            yp_r = backtransform_sigma(yp, sigma2 or 0.0).numpy()
        yp_r = apply_headroom_clamp(yp_r, cap)

        mic_real = evaluate_regression(yp_r, yt_r)
        mic_r2t_real = trimmed_r2(yt_r, yp_r, trim=0.01)

        micro = {
            "micro_MAE_log": mic_log["MAE"],
            "micro_RMSE_log": mic_log["RMSE"],
            "micro_R2_log": mic_log["R2"],
            "micro_R2trim_log": mic_r2t_log,
            "micro_MAE_real": mic_real["MAE"],
            "micro_RMSE_real": mic_real["RMSE"],
            "micro_SMAPE_real": mic_real["SMAPE"],
            "micro_R2_real": mic_real["R2"],
            "micro_R2trim_real": mic_r2t_real,
        }

    return avg_loss, weekly_rows, micro


# ---------------------- ước lượng thống kê TRAIN ----------------------
@torch.no_grad()
def estimate_backtransform_stats_on_train(model, edge_index, train_paths) -> dict:
    resids = []
    y_train_real = []

    model.eval()
    for p in train_paths:
        d = _safe_load(p)
        x = d["x"].to(DEVICE)
        y = d["y"].to(DEVICE)

        mask = d.get("train_mask", d.get("mask_train", None))
        if mask is None or mask.sum().item() == 0:
            continue

        feature_cols = d.get("feature_cols", [])
        temporal_seq, _ = build_temporal_seq(x, feature_cols)
        y_hat = model(x, edge_index, temporal_seq=temporal_seq)

        if y_hat.dim() == 2 and y_hat.size(-1) == 1:
            y_hat = y_hat.squeeze(-1)
        if y.dim() == 2 and y.size(-1) == 1:
            y = y.squeeze(-1)

        r = (y[mask] - y_hat[mask]).detach().cpu().numpy()
        if r.size > 0:
            resids.append(r)

        y_train_real.append(torch.expm1(y[mask]).clamp_min(0).cpu().numpy())

    if not resids:
        return {"smear": None, "sigma2": 0.0, "cap": None}

    r = np.concatenate(resids, axis=0)
    sigma2 = float(max(0.0, np.var(r, ddof=1)))
    smear = float(np.mean(np.exp(r)))

    cap = None
    if y_train_real:
        ytr = np.concatenate(y_train_real, axis=0)
        cap = float(np.quantile(ytr, 0.999))

    return {"smear": smear, "sigma2": sigma2, "cap": cap}


def compute_val_loss_after_load_best(model, edge_index, va_paths, loss_fn) -> float:
    val_loss, _, _ = run_epoch(
        model, edge_index, va_paths,
        phase="val", opt=None, loss_fn=loss_fn
    )
    return float(val_loss)


# ---------------------- main ----------------------
def main():
    seed_all(42)

    paths = list_snapshots()
    tr_paths, va_paths, te_paths = split_weeks(paths)

    sample = _safe_load(tr_paths[0])
    in_dim = int(sample["x"].shape[1])
    has_temporal = bool(_find_lag_columns(sample.get("feature_cols", [])))
    edge_index = _load_edge(EDGE_PATH).to(DEVICE)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gat",
                        choices=["gat", "gcn", "gnn", "sheaf", "sheaf_conn"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval_only", type=int, default=0,
                        help="1: skip train, load best ckpt and eval")
    parser.add_argument("--export_predictions", type=int, default=0,
                        help="1: export node_predictions at final eval only")
    args = parser.parse_args()

    # output theo model
    best_ckpt = os.path.join(CKPT_DIR, f"{args.model}_global_best.pt")
    weekly_csv = os.path.join(REPORT_DIR, f"{args.model}_global_weekly_report.csv")
    summary_json = os.path.join(REPORT_DIR, f"{args.model}_global_summary.json")
    epoch_log_csv = os.path.join(REPORT_DIR, f"{args.model}_epoch_log.csv")

    # predictions theo model + tách folder
    model_vis_dir = os.path.join(VIS_DATA_DIR, args.model)
    os.makedirs(model_vis_dir, exist_ok=True)
    predictions_csv = os.path.join(model_vis_dir, f"node_predictions_{args.model}.csv")

    # ---------------------- chọn model + hyperparams riêng ----------------------
    if args.model == "gat":
        hidden_dim = 128
        lr = 3e-4
        wd = 1e-4
        model = make_model(
            in_dim=in_dim,
            has_temporal=has_temporal,
            hidden=hidden_dim,
            heads=(4, 4),
            gat_dropout=0.2
        ).to(DEVICE)

    elif args.model == "sheaf_conn":
        hidden_dim = 64
        lr = 1e-4
        wd = 5e-4
        model = build_model(
            args.model,
            in_dim=in_dim,
            hidden=hidden_dim,
            out_dim=1,
            dropout=0.2
        ).to(DEVICE)

    else:
        hidden_dim = 128
        lr = 3e-4
        wd = 1e-4
        model = build_model(
            args.model,
            in_dim=in_dim,
            hidden=hidden_dim,
            out_dim=1,
            dropout=0.2
        ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.HuberLoss(delta=1.2)

    patience, bad = 30, 0
    EPOCHS = int(args.epochs)
    best_val = float("inf")

    print(f"Selected model: {args.model}")
    print(f"Training epochs: {args.epochs}")
    print(f"Global training with {len(tr_paths)} train weeks, {len(va_paths)} val weeks, {len(te_paths)} test weeks")
    print(f"Device: {DEVICE} | Model={args.model} | in={in_dim}, hidden={hidden_dim}, lr={lr}, wd={wd}, temporal={'ON' if has_temporal else 'OFF'}")

    t0 = time.time()

    # reset epoch log nếu train mới
    if int(args.eval_only) != 1:
        if os.path.exists(epoch_log_csv):
            os.remove(epoch_log_csv)

    # ---------------------- train ----------------------
    if int(args.eval_only) != 1:
        for ep in range(1, EPOCHS + 1):
            tr_loss, _, _ = run_epoch(model, edge_index, tr_paths, phase="train", opt=opt, loss_fn=loss_fn)
            va_loss, _, _ = run_epoch(model, edge_index, va_paths, phase="val", opt=None, loss_fn=loss_fn)

            elapsed_min = (time.time() - t0) / 60.0

            # log hội tụ (paper): 1..10 và mỗi 10 epoch
            if (ep <= 10) or (ep % 10 == 0) or (ep == EPOCHS):
                print(f"Epoch {ep:03d} | Train {tr_loss:.4f} | Val {va_loss:.4f}")

                append_epoch_log(epoch_log_csv, {
                    "model": args.model,
                    "epoch": ep,
                    "train_loss": float(tr_loss),
                    "val_loss": float(va_loss),
                    "best_val_so_far": float(min(best_val, va_loss)),
                    "elapsed_min": float(elapsed_min)
                })

            if va_loss + 1e-9 < best_val:
                best_val = va_loss
                bad = 0
                torch.save({"state_dict": model.state_dict(), "in_dim": in_dim, "has_temporal": has_temporal}, best_ckpt)
            else:
                bad += 1
                if bad >= patience:
                    print(f"⏹️ Early stopping @ epoch {ep}")
                    break
    else:
        if not os.path.exists(best_ckpt):
            raise FileNotFoundError(f"Không tìm thấy checkpoint best: {best_ckpt}")
        print("⚠️ eval_only=1: Skip training, load best checkpoint.")

    # ---------------------- load best & stats ----------------------
    ckpt = _safe_load(best_ckpt)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    model.eval()

    if int(args.eval_only) == 1:
        best_val = compute_val_loss_after_load_best(model, edge_index, va_paths, loss_fn)

    stats = estimate_backtransform_stats_on_train(model, edge_index, tr_paths)
    smear, sigma2, cap = stats["smear"], stats["sigma2"], stats["cap"]

    # ---------------------- eval cuối ----------------------
    do_export_pred = (int(args.export_predictions) == 1)

    # nếu export predictions -> ghi đè file của model để không phình
    if do_export_pred and os.path.exists(predictions_csv):
        os.remove(predictions_csv)

    val_loss_final, val_rows, val_micro = run_epoch(
        model, edge_index, va_paths,
        phase="val", opt=None, loss_fn=loss_fn,
        smear=smear, sigma2=sigma2, cap=cap,
        write_predictions=do_export_pred, predictions_csv=predictions_csv
    )

    test_loss_final, test_rows, test_micro = run_epoch(
        model, edge_index, te_paths,
        phase="test", opt=None, loss_fn=loss_fn,
        smear=smear, sigma2=sigma2, cap=cap,
        write_predictions=do_export_pred, predictions_csv=predictions_csv
    )

    # ---------------------- weekly report ----------------------
    weekly_rows = val_rows + test_rows
    columns = [
        "Year", "Epiweek", "Split",
        "MAE_log", "RMSE_log", "R2_log", "R2trim_log",
        "MAE_real", "RMSE_real", "SMAPE_real", "R2_real", "R2trim_real"
    ]
    weekly_df = pd.DataFrame(weekly_rows, columns=columns)
    weekly_df.to_csv(weekly_csv, index=False)

    def macro_avg(df: pd.DataFrame, split: str) -> dict:
        sub = df[df["Split"] == split]
        if sub.empty:
            return {}
        keys = [
            "MAE_log", "RMSE_log", "R2_log", "R2trim_log",
            "MAE_real", "RMSE_real", "SMAPE_real", "R2_real", "R2trim_real"
        ]
        return {f"{split}_macro_{k}": float(sub[k].mean()) for k in keys}

    macro_val = macro_avg(weekly_df, "val")
    macro_test = macro_avg(weekly_df, "test")

    def cls_metrics(paths, phase: str) -> dict:
        yt_all, yp_all = [], []
        for p in paths:
            d = _safe_load(p)
            x = d["x"].to(DEVICE)
            y = d["y"].to(DEVICE)
            mask = d.get(f"{phase}_mask", d.get(f"mask_{phase}", None))
            if mask is None or mask.sum().item() == 0:
                continue

            feature_cols = d.get("feature_cols", [])
            temporal_seq, _ = build_temporal_seq(x, feature_cols)
            yp = model(x, edge_index, temporal_seq=temporal_seq)

            if yp.dim() == 2 and yp.size(-1) == 1:
                yp = yp.squeeze(-1)
            if y.dim() == 2 and y.size(-1) == 1:
                y = y.squeeze(-1)

            yt_all.append(y[mask].detach().cpu().view(-1))
            yp_all.append(yp[mask].detach().cpu().view(-1))

        if not yt_all:
            return {}

        yt = torch.cat(yt_all, 0)
        yp = torch.cat(yp_all, 0)

        yt_r = torch.expm1(yt).clamp_min(0).numpy()
        if smear is not None and smear > 0:
            yp_r = backtransform_smear(yp, smear).numpy()
        else:
            yp_r = backtransform_sigma(yp, sigma2 or 0.0).numpy()
        yp_r = apply_headroom_clamp(yp_r, cap)

        cls = classification_metrics_from_regression(yt_r, yp_r, pos_threshold=None, q=0.90)
        return {f"{phase}_ROC_AUC": cls["ROC_AUC"], f"{phase}_PR_AUC": cls["PR_AUC"], f"{phase}_pos_rate": cls["pos_rate"]}

    cls_val = cls_metrics(va_paths, "val")
    cls_test = cls_metrics(te_paths, "test")

    summary = {
        "best_val_loss": float(best_val),
        "val_loss_final": float(val_loss_final),
        "test_loss_final": float(test_loss_final),
        "smear_train": None if smear is None else float(smear),
        "sigma2_train_log": float(sigma2),
        "cap_train_q999": None if cap is None else float(cap),
        **macro_val, **(val_micro or {}), **cls_val,
        **macro_test, **(test_micro or {}), **cls_test,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved best checkpoint: {best_ckpt}")
    print(f"✅ Wrote weekly report:  {weekly_csv}")
    print(f"✅ Wrote epoch log:      {epoch_log_csv}")
    if do_export_pred:
        print(f"✅ Wrote node-level predictions: {predictions_csv}")
    else:
        print("ℹ️ Node-level predictions: OFF (use --export_predictions 1 to export).")
    print(f"✅ Summary saved: {summary_json}")
    print(f"⏱️ Done in {(time.time() - t0)/60:.2f} min.")


if __name__ == "__main__":
    main()