import os
import glob
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interim_dir", type=str, default="data/interim")
    parser.add_argument("--pattern", type=str, default="*_epoch_log.csv")
    parser.add_argument("--out_dir", type=str, default="visualizations/plots")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Chỉ plot các model này (vd: gnn gcn gat sheaf sheaf_conn). Nếu bỏ trống -> lấy hết.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.interim_dir, args.pattern)))
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy epoch log: {args.interim_dir}/{args.pattern}")

    wanted = None
    if args.models:
        wanted = set([m.lower() for m in args.models])

    all_logs = []
    for p in paths:
        model = os.path.basename(p).replace("_epoch_log.csv", "")
        if wanted and (model not in wanted):
            continue

        df = pd.read_csv(p)
        if "model" not in df.columns:
            df["model"] = model
        else:
            df["model"] = df["model"].fillna(model)
        all_logs.append(df)

    if not all_logs:
        raise RuntimeError("Không có epoch_log nào được chọn để plot.")

    logs = pd.concat(all_logs, ignore_index=True)

    # ----- Plot ALL: val_loss -----
    plt.figure()
    for model, g in logs.groupby("model"):
        g = g.sort_values("epoch")
        plt.plot(g["epoch"], g["val_loss"], label=model)
    plt.xlabel("Epoch")
    plt.ylabel("Val Loss")
    plt.title("Convergence (Val Loss)")
    plt.legend()
    out1 = os.path.join(args.out_dir, "convergence_val_loss_all.png")
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out1}")

    # ----- Plot ALL: train_loss -----
    plt.figure()
    for model, g in logs.groupby("model"):
        g = g.sort_values("epoch")
        plt.plot(g["epoch"], g["train_loss"], label=model)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Convergence (Train Loss)")
    plt.legend()
    out2 = os.path.join(args.out_dir, "convergence_train_loss_all.png")
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out2}")

    # ----- Plot per model: train vs val -----
    for model, g in logs.groupby("model"):
        g = g.sort_values("epoch")
        plt.figure()
        plt.plot(g["epoch"], g["train_loss"], label="train_loss")
        plt.plot(g["epoch"], g["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Convergence: {model}")
        plt.legend()
        outm = os.path.join(args.out_dir, f"convergence_{model}_train_val.png")
        plt.savefig(outm, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {outm}")


if __name__ == "__main__":
    main()