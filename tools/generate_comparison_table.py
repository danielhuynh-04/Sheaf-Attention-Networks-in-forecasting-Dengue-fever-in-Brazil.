import os
import glob
import json
import argparse
from typing import Any, Dict, List

import pandas as pd


DEFAULT_KEYS = [
    # loss
    "best_val_loss", "val_loss_final", "test_loss_final",
    # val macro (log)
    "val_macro_MAE_log", "val_macro_RMSE_log", "val_macro_R2_log", "val_macro_R2trim_log",
    # test macro (log)
    "test_macro_MAE_log", "test_macro_RMSE_log", "test_macro_R2_log", "test_macro_R2trim_log",
    # val macro (real)
    "val_macro_MAE_real", "val_macro_RMSE_real", "val_macro_SMAPE_real", "val_macro_R2_real", "val_macro_R2trim_real",
    # test macro (real)
    "test_macro_MAE_real", "test_macro_RMSE_real", "test_macro_SMAPE_real", "test_macro_R2_real", "test_macro_R2trim_real",
    # micro (log)
    "micro_MAE_log", "micro_RMSE_log", "micro_R2_log", "micro_R2trim_log",
    # micro (real)
    "micro_MAE_real", "micro_RMSE_real", "micro_SMAPE_real", "micro_R2_real", "micro_R2trim_real",
    # AUC/PR
    "val_ROC_AUC", "val_PR_AUC", "val_pos_rate",
    "test_ROC_AUC", "test_PR_AUC", "test_pos_rate",
    # train stats for backtransform
    "smear_train", "sigma2_train_log", "cap_train_q999",
]


def safe_get(d: Dict[str, Any], k: str):
    v = d.get(k, None)
    try:
        if isinstance(v, float) and (pd.isna(v) or v != v):
            return None
    except Exception:
        return None
    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interim_dir", type=str, default="data/interim")
    parser.add_argument("--pattern", type=str, default="*_global_summary.json")
    parser.add_argument("--out_csv", type=str, default="data/interim/model_comparison.csv")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Chỉ lấy các model này (vd: gnn gcn gat sheaf sheaf_conn). Nếu bỏ trống -> lấy hết.")
    parser.add_argument("--keys", nargs="*", default=None,
                        help="Override danh sách keys. Nếu bỏ trống -> dùng DEFAULT_KEYS.")
    args = parser.parse_args()

    keys = args.keys if args.keys else DEFAULT_KEYS

    paths = sorted(glob.glob(os.path.join(args.interim_dir, args.pattern)))
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy file summary theo pattern: {args.interim_dir}/{args.pattern}")

    wanted = None
    if args.models:
        wanted = set([m.lower() for m in args.models])

    rows: List[Dict[str, Any]] = []
    for p in paths:
        name = os.path.basename(p).replace("_global_summary.json", "")
        if wanted and (name not in wanted):
            continue

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        row = {"model": name}
        for k in keys:
            row[k] = safe_get(data, k)
        rows.append(row)

    if not rows:
        raise RuntimeError("Không có model nào được chọn để gom bảng.")

    df = pd.DataFrame(rows)

    # sort: ưu tiên test_macro_R2_log giảm dần nếu có, không thì theo model
    sort_key = "test_macro_R2_log" if "test_macro_R2_log" in df.columns else None
    if sort_key and df[sort_key].notna().any():
        df = df.sort_values(by=sort_key, ascending=False)
    else:
        df = df.sort_values(by="model", ascending=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"✅ Wrote comparison table: {args.out_csv}")
    print(df)


if __name__ == "__main__":
    main()