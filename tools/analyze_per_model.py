import os
import glob
import json
import argparse
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SUMMARY_SUFFIX = "_global_summary.json"
WEEKLY_SUFFIX = "_global_weekly_report.csv"
EPOCH_SUFFIX = "_epoch_log.csv"


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def plot_weekly_metric(df: pd.DataFrame, metric: str, title: str, out_path: str):
    if metric not in df.columns:
        return

    plt.figure()
    for split, g in df.groupby("Split"):
        g = g.sort_values(["Year", "Epiweek"])
        x = g["Year"].astype(int) * 100 + g["Epiweek"].astype(int)
        plt.plot(x.values, g[metric].values, label=split)

    plt.xlabel("Year*100 + Epiweek")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_convergence(epoch_df: pd.DataFrame, model: str, out_path: str):
    if "epoch" not in epoch_df.columns:
        return
    if ("train_loss" not in epoch_df.columns) or ("val_loss" not in epoch_df.columns):
        return

    g = epoch_df.sort_values("epoch")
    plt.figure()
    plt.plot(g["epoch"].values, g["train_loss"].values, label="train_loss")
    plt.plot(g["epoch"].values, g["val_loss"].values, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Convergence: {model}")
    plt.legend()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def weekly_extremes(df: pd.DataFrame, metric: str, k: int = 10) -> pd.DataFrame:
    if metric not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    is_higher_better = metric.lower().startswith("r2")

    out_rows = []
    for split, g in df.groupby("Split"):
        g = g.dropna(subset=[metric]).copy()
        if g.empty:
            continue

        if is_higher_better:
            best = g.sort_values(metric, ascending=False).head(k)
            worst = g.sort_values(metric, ascending=True).head(k)
        else:
            best = g.sort_values(metric, ascending=True).head(k)
            worst = g.sort_values(metric, ascending=False).head(k)

        best = best.assign(rank_type="best")
        worst = worst.assign(rank_type="worst")
        out_rows.append(best)
        out_rows.append(worst)

    if not out_rows:
        return pd.DataFrame()

    out = pd.concat(out_rows, ignore_index=True)
    cols_front = ["Split", "rank_type", "Year", "Epiweek", metric]
    cols_front = [c for c in cols_front if c in out.columns]
    rest = [c for c in out.columns if c not in cols_front]
    return out[cols_front + rest]


def summarize_weekly_stats(df: pd.DataFrame, prefix: str) -> Dict[str, Any]:
    metrics = [
        "MAE_log", "RMSE_log", "R2_log", "R2trim_log",
        "MAE_real", "RMSE_real", "SMAPE_real", "R2_real", "R2trim_real",
    ]
    out: Dict[str, Any] = {}

    for split, g in df.groupby("Split"):
        for m in metrics:
            if m not in g.columns:
                continue
            s = g[m].dropna()
            if s.empty:
                continue
            out[f"{prefix}_{split}_{m}_mean"] = float(s.mean())
            out[f"{prefix}_{split}_{m}_std"] = float(s.std(ddof=1)) if s.size > 1 else 0.0
            out[f"{prefix}_{split}_{m}_median"] = float(s.median())

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interim_dir", type=str, default="data/interim")
    parser.add_argument("--out_dir", type=str, default="visualizations/plots")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Chỉ phân tích các model này (vd: gnn gcn gat sheaf sheaf_conn). Nếu bỏ trống -> lấy hết.")
    parser.add_argument("--topk", type=int, default=10, help="Top-k tuần tốt/xấu để xuất")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(args.interim_dir)

    summary_paths = sorted(glob.glob(os.path.join(args.interim_dir, f"*{SUMMARY_SUFFIX}")))
    if not summary_paths:
        raise FileNotFoundError(f"Không tìm thấy *{SUMMARY_SUFFIX} trong {args.interim_dir}")

    wanted = None
    if args.models:
        wanted = set([m.lower() for m in args.models])

    detailed_rows: List[Dict[str, Any]] = []

    for sp in summary_paths:
        model = os.path.basename(sp).replace(SUMMARY_SUFFIX, "")
        if wanted and (model not in wanted):
            continue

        weekly_path = os.path.join(args.interim_dir, f"{model}{WEEKLY_SUFFIX}")
        epoch_path = os.path.join(args.interim_dir, f"{model}{EPOCH_SUFFIX}")

        summary = read_json(sp)

        weekly_df: Optional[pd.DataFrame] = None
        if os.path.exists(weekly_path):
            weekly_df = pd.read_csv(weekly_path)
        else:
            print(f"⚠️ Missing weekly report for {model}: {weekly_path}")

        epoch_df: Optional[pd.DataFrame] = None
        if os.path.exists(epoch_path):
            epoch_df = pd.read_csv(epoch_path)

        # ---- weekly plots ----
        if weekly_df is not None and not weekly_df.empty:
            for metric in ["R2_log", "MAE_log", "RMSE_log", "R2_real", "MAE_real", "RMSE_real"]:
                plot_weekly_metric(
                    weekly_df,
                    metric,
                    title=f"{model} Weekly {metric} (val/test)",
                    out_path=os.path.join(args.out_dir, f"weekly_{metric}_{model}.png")
                )

            # ---- extremes tables ----
            for m in ["R2_log", "MAE_log", "R2_real", "MAE_real"]:
                ext = weekly_extremes(weekly_df, m, k=int(args.topk))
                if not ext.empty:
                    ext_path = os.path.join(args.interim_dir, f"{model}_week_extremes_{m}.csv")
                    ext.to_csv(ext_path, index=False, encoding="utf-8")

        # ---- convergence plot ----
        if epoch_df is not None and not epoch_df.empty:
            plot_convergence(
                epoch_df,
                model=model,
                out_path=os.path.join(args.out_dir, f"convergence_{model}_train_val.png")
            )

        # ---- per-model detailed stats row ----
        row: Dict[str, Any] = {"model": model}

        core_keys = [
            "best_val_loss", "val_loss_final", "test_loss_final",
            "val_macro_MAE_log", "val_macro_RMSE_log", "val_macro_R2_log",
            "test_macro_MAE_log", "test_macro_RMSE_log", "test_macro_R2_log",
            "val_macro_MAE_real", "val_macro_RMSE_real", "val_macro_R2_real",
            "test_macro_MAE_real", "test_macro_RMSE_real", "test_macro_R2_real",
            "val_PR_AUC", "test_PR_AUC", "val_ROC_AUC", "test_ROC_AUC",
        ]
        for k in core_keys:
            row[k] = safe_float(summary.get(k, None))

        if weekly_df is not None and not weekly_df.empty:
            row.update(summarize_weekly_stats(weekly_df, prefix="weekly"))

        if epoch_df is not None and not epoch_df.empty:
            e = epoch_df.sort_values("epoch")
            row["epoch_log_last_epoch"] = int(e["epoch"].iloc[-1])
            row["epoch_log_last_train_loss"] = safe_float(e["train_loss"].iloc[-1])
            row["epoch_log_last_val_loss"] = safe_float(e["val_loss"].iloc[-1])
            row["epoch_log_best_val_loss"] = safe_float(e["val_loss"].min())
            if "elapsed_min" in e.columns:
                row["epoch_log_last_elapsed_min"] = safe_float(e["elapsed_min"].iloc[-1])

        detailed_rows.append(row)

    out_csv = os.path.join(args.interim_dir, "per_model_detailed_stats.csv")
    df_out = pd.DataFrame(detailed_rows)

    if "test_macro_R2_log" in df_out.columns and df_out["test_macro_R2_log"].notna().any():
        df_out = df_out.sort_values("test_macro_R2_log", ascending=False)

    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✅ Wrote detailed per-model stats: {out_csv}")
    print(f"✅ Plots saved under: {args.out_dir}")


if __name__ == "__main__":
    main()