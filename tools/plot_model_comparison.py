import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bar_group_plot(df, models, metrics, title, ylabel, out_path):
    x = np.arange(len(models))
    width = 0.8 / max(1, len(metrics))

    plt.figure()
    for i, m in enumerate(metrics):
        vals = df.set_index("model").reindex(models)[m].astype(float).values
        plt.bar(x + i * width, vals, width, label=m)

    plt.xticks(x + width * (len(metrics) - 1) / 2, models, rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/interim/model_comparison.csv")
    parser.add_argument("--out_dir", type=str, default="visualizations/plots")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Chỉ plot các model này theo thứ tự mong muốn (vd: gnn gcn gat sheaf sheaf_conn). Nếu bỏ trống -> dùng thứ tự trong CSV.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "model" not in df.columns:
        raise ValueError("CSV phải có cột 'model'.")

    if args.models:
        models = [m.lower() for m in args.models]
    else:
        models = df["model"].tolist()

    needed = [
        "val_macro_MAE_log", "val_macro_RMSE_log", "val_macro_R2_log",
        "test_macro_MAE_log", "test_macro_RMSE_log", "test_macro_R2_log",
        "val_macro_MAE_real", "val_macro_RMSE_real", "val_macro_R2_real",
        "test_macro_MAE_real", "test_macro_RMSE_real", "test_macro_R2_real",
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    out1 = os.path.join(args.out_dir, "model_compare_val_macro_log.png")
    bar_group_plot(
        df, models,
        metrics=["val_macro_MAE_log", "val_macro_RMSE_log", "val_macro_R2_log"],
        title="Model Comparison (VAL macro, log-space)",
        ylabel="Metric value",
        out_path=out1
    )
    print(f"✅ Saved: {out1}")

    out2 = os.path.join(args.out_dir, "model_compare_test_macro_log.png")
    bar_group_plot(
        df, models,
        metrics=["test_macro_MAE_log", "test_macro_RMSE_log", "test_macro_R2_log"],
        title="Model Comparison (TEST macro, log-space)",
        ylabel="Metric value",
        out_path=out2
    )
    print(f"✅ Saved: {out2}")

    out3 = os.path.join(args.out_dir, "model_compare_val_macro_real.png")
    bar_group_plot(
        df, models,
        metrics=["val_macro_MAE_real", "val_macro_RMSE_real", "val_macro_R2_real"],
        title="Model Comparison (VAL macro, real-space)",
        ylabel="Metric value",
        out_path=out3
    )
    print(f"✅ Saved: {out3}")

    out4 = os.path.join(args.out_dir, "model_compare_test_macro_real.png")
    bar_group_plot(
        df, models,
        metrics=["test_macro_MAE_real", "test_macro_RMSE_real", "test_macro_R2_real"],
        title="Model Comparison (TEST macro, real-space)",
        ylabel="Metric value",
        out_path=out4
    )
    print(f"✅ Saved: {out4}")


if __name__ == "__main__":
    main()