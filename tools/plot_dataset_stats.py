# tools/plot_dataset_stats.py
import os
import pandas as pd
import matplotlib.pyplot as plt

INPUT_BY_SPLIT = "data/interim/dataset_stats_by_split.csv"
INPUT_BY_YEAR  = "data/interim/dataset_stats_by_year.csv"
OUT_DIR = "visualizations/figures"

def savefig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    by_split = pd.read_csv(INPUT_BY_SPLIT)
    by_year  = pd.read_csv(INPUT_BY_YEAR)

    # ---- 1) Zero-rate by split ----
    if "y_real_zero_rate" in by_split.columns:
        plt.figure()
        plt.bar(by_split["split"], by_split["y_real_zero_rate"])
        plt.xlabel("Split")
        plt.ylabel("Zero rate (y_real = 0)")
        plt.title("Zero-inflation by split")
        savefig(os.path.join(OUT_DIR, "fig_dataset_split_zero_rate.png"))

    # ---- 2) Zero-rate by year ----
    if "y_real_zero_rate" in by_year.columns:
        plt.figure()
        plt.plot(by_year["year"], by_year["y_real_zero_rate"], marker="o")
        plt.xlabel("Year")
        plt.ylabel("Zero rate (y_real = 0)")
        plt.title("Zero-inflation over years")
        savefig(os.path.join(OUT_DIR, "fig_dataset_year_zero_rate.png"))

    # ---- 3) y_real_max by year ----
    if "y_real_max" in by_year.columns:
        plt.figure()
        plt.plot(by_year["year"], by_year["y_real_max"], marker="o")
        plt.xlabel("Year")
        plt.ylabel("Max weekly cases (y_real_max)")
        plt.title("Year-wise extremes (max weekly cases)")
        savefig(os.path.join(OUT_DIR, "fig_dataset_year_ymax.png"))

    # ---- 4) y_real_p99 by year ----
    if "y_real_p99" in by_year.columns:
        plt.figure()
        plt.plot(by_year["year"], by_year["y_real_p99"], marker="o")
        plt.xlabel("Year")
        plt.ylabel("P99 weekly cases (y_real_p99)")
        plt.title("Year-wise tail (P99 weekly cases)")
        savefig(os.path.join(OUT_DIR, "fig_dataset_year_p99.png"))

    # ---- 5) P99 by split (real-space) ----
    if "y_real_p99" in by_split.columns:
        plt.figure()
        plt.bar(by_split["split"], by_split["y_real_p99"])
        plt.xlabel("Split")
        plt.ylabel("P99 weekly cases (y_real_p99)")
        plt.title("Tail severity by split (P99)")
        savefig(os.path.join(OUT_DIR, "fig_dataset_split_p99.png"))

    print(f"✅ Figures saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()