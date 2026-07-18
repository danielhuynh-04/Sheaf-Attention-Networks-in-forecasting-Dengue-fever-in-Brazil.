# tools/run_benchmark.py
# ---------------------------------------
# Wrapper chạy nhiều model tuần tự (paper-safe)
# - Train + eval để sinh: summary.json, weekly_report.csv, epoch_log.csv
# - Không export node_predictions mặc định (tránh tràn ổ)
# - Nếu cần export, chỉ export 1 model (default: gat) sau khi train xong
# ---------------------------------------

import subprocess
import argparse
import sys


def run_cmd(cmd):
    print(" ".join(cmd))
    return subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Số epoch cho mỗi model"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["gnn", "gcn", "gat", "sheaf_conn", "sheaf"],
        help="Danh sách model muốn chạy theo thứ tự"
    )

    parser.add_argument(
        "--continue_on_fail",
        type=int,
        default=0,
        help="1: model lỗi vẫn chạy tiếp model sau"
    )

    parser.add_argument(
        "--export_predictions",
        type=int,
        default=0,
        help="1: export node_predictions cho 1 model sau khi train xong"
    )

    parser.add_argument(
        "--export_model",
        type=str,
        default="gat",
        help="Model sẽ export node_predictions (mặc định: gat)"
    )

    args = parser.parse_args()

    models = [m.lower() for m in args.models]

    print(f"\n🔁 Running benchmark with {args.epochs} epochs per model")
    print(f"Models order: {models}")
    print(f"Export predictions: {args.export_predictions} | export_model: {args.export_model}\n")

    # 1) Train/eval cho từng model (không export predictions)
    for m in models:
        print("\n==============================")
        print(f"Running model: {m}")
        print("==============================\n")

        cmd = [
            sys.executable,          # dùng đúng python đang chạy
            "run_global_gat.py",
            "--model", m,
            "--epochs", str(args.epochs),
            "--export_predictions", "0"
        ]

        result = run_cmd(cmd)

        if result.returncode != 0:
            print(f"❌ Model {m} failed with code {result.returncode}.")
            if int(args.continue_on_fail) != 1:
                print("\n⛔ Stopping benchmark (set --continue_on_fail 1 to continue).")
                break

    # 2) Export node_predictions (chỉ 1 model) nếu cần
    if int(args.export_predictions) == 1:
        export_model = args.export_model.lower()
        if export_model not in models:
            print(f"\n⚠️ export_model '{export_model}' không nằm trong danh sách models đã chạy.")
            print("   Bạn vẫn có thể export nếu model đó đã có checkpoint best trước đó.")
        print("\n==============================")
        print(f"Exporting node predictions for: {export_model}")
        print("==============================\n")

        cmd = [
            sys.executable,
            "run_global_gat.py",
            "--model", export_model,
            "--eval_only", "1",
            "--export_predictions", "1"
        ]
        run_cmd(cmd)

    print("\n✅ Benchmark finished.")


if __name__ == "__main__":
    main()