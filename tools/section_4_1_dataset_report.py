# tools/section_4_1_dataset_report.py
from __future__ import annotations
import os, glob, json
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# ==============================
# CONFIG (chỉnh nếu bạn đổi path)
# ==============================
SNAP_DIR = "data/processed/weekly_pt_scaled"
EDGE_PATH = "data/processed/edge_index.pt"
NODE2IDX_PATH = "data/processed/node2idx.json"
SCALER_PATH = "data/processed/scaler_weekly.json"

STATS_OVERALL_JSON = "data/interim/dataset_stats_overall.json"
STATS_BY_YEAR_CSV = "data/interim/dataset_stats_by_year.csv"
STATS_BY_SPLIT_CSV = "data/interim/dataset_stats_by_split.csv"
FEATURE_COLS_JSON = "data/interim/dataset_stats_feature_cols.json"  # optional

OUT_XLSX = "data/interim/4_1_dataset_section.xlsx"
OUT_TABLE_CSV = "data/interim/4_1_table_dataset_summary.csv"
FIG_DIR = "visualizations/section_4_1"

os.makedirs(os.path.dirname(OUT_XLSX), exist_ok=True)
os.makedirs(os.path.dirname(OUT_TABLE_CSV), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ==============================
# UTILS
# ==============================
def _safe_load_pt(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

def list_snapshots() -> List[str]:
    paths = sorted(glob.glob(os.path.join(SNAP_DIR, "*.pt")))
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy *.pt trong: {SNAP_DIR}")
    return paths

def parse_year_epiweek(pt_path: str) -> Tuple[int, int]:
    base = os.path.basename(pt_path).replace(".pt", "")
    y, w = base.split("_")
    return int(y), int(w)

def as_1d_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.astype(float).reshape(-1)

def load_json_if_exists(path: str) -> Dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_edge_index(path: str) -> torch.Tensor:
    ei = _safe_load_pt(path)
    # hỗ trợ trường hợp lưu dict
    if isinstance(ei, dict) and "edge_index" in ei:
        ei = ei["edge_index"]
    if not torch.is_tensor(ei):
        raise ValueError("edge_index.pt phải là torch.Tensor (hoặc dict chứa 'edge_index').")
    if ei.dim() != 2 or ei.size(0) != 2:
        raise ValueError(f"edge_index phải có shape [2,E], nhận {tuple(ei.shape)}")
    return ei.long()

def robust_quantiles(arr: np.ndarray, qs=(0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)) -> Dict[str, float]:
    out = {}
    for q in qs:
        out[f"q{int(q*100):02d}"] = float(np.quantile(arr, q))
    return out

def save_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ==============================
# CORE: build tables for Section 4.1
# ==============================
def compute_graph_stats(edge_index: torch.Tensor, N: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    E = int(edge_index.size(1))

    self_loops = int((src == dst).sum())

    # degree (out-degree theo src) - đủ để mô tả sparsity
    deg = np.bincount(src, minlength=N)

    graph_overview = {
        "num_nodes_N": N,
        "num_edges_E": E,
        "self_loops": self_loops,
        "avg_degree_out": float(deg.mean()),
        "median_degree_out": float(np.median(deg)),
        "max_degree_out": int(deg.max()),
        "isolated_nodes_deg0": int((deg == 0).sum()),
        "edge_index_format": "[2, E] (PyTorch Geometric)",
    }
    graph_overview_df = pd.DataFrame([graph_overview])

    deg_stats = {
        "deg_min": int(deg.min()),
        "deg_mean": float(deg.mean()),
        "deg_std": float(deg.std(ddof=1)) if N > 1 else 0.0,
        "deg_median": float(np.median(deg)),
        "deg_p90": float(np.quantile(deg, 0.90)),
        "deg_p95": float(np.quantile(deg, 0.95)),
        "deg_p99": float(np.quantile(deg, 0.99)),
        "deg_max": int(deg.max()),
        "deg0_count": int((deg == 0).sum()),
    }
    deg_stats_df = pd.DataFrame(list(deg_stats.items()), columns=["metric", "value"])

    return graph_overview_df, deg_stats_df

def compute_time_coverage(paths: List[str]) -> pd.DataFrame:
    rows = []
    for p in paths:
        y, w = parse_year_epiweek(p)
        rows.append((y, w, os.path.basename(p)))
    df = pd.DataFrame(rows, columns=["year", "epiweek", "file"])
    df = df.sort_values(["year", "epiweek"]).reset_index(drop=True)

    # per-year missing weeks (basic within min..max)
    missing = []
    for y in sorted(df["year"].unique()):
        weeks = df.loc[df["year"] == y, "epiweek"].tolist()
        if not weeks:
            continue
        wmin, wmax = int(min(weeks)), int(max(weeks))
        expected = set(range(wmin, wmax + 1))
        miss = sorted(list(expected - set(map(int, weeks))))
        for mw in miss:
            missing.append({"year": y, "missing_epiweek": mw})

    cov = {
        "years_start": int(df["year"].min()),
        "years_end": int(df["year"].max()),
        "snapshots_total": int(len(df)),
        "note": "Missing weeks listed separately (if any)."
    }
    cov_df = pd.DataFrame([cov])
    missing_df = pd.DataFrame(missing)

    return cov_df, missing_df, df[["year", "epiweek"]]

def scan_snapshot_schema(paths: List[str], samples: int = 3) -> pd.DataFrame:
    rows = []
    for p in paths[:samples]:
        d = _safe_load_pt(p)
        x = d.get("x", None)
        y = d.get("y", None)
        rows.append({
            "sample_file": os.path.basename(p),
            "keys": ", ".join(sorted(list(d.keys()))),
            "x_shape": str(tuple(x.shape)) if torch.is_tensor(x) else None,
            "y_shape": str(tuple(y.shape)) if torch.is_tensor(y) else None,
            "has_feature_cols": "feature_cols" in d,
            "has_label_transform": "label_transform" in d,
            "has_geocodes": "geocodes" in d,
            "has_masks": all(k in d for k in ["train_mask", "val_mask", "test_mask"]),
        })
    return pd.DataFrame(rows)

def build_feature_tables(paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d0 = _safe_load_pt(paths[0])
    feature_cols = d0.get("feature_cols", None)
    label_transform = d0.get("label_transform", None)

    feat_meta = pd.DataFrame([{
        "num_features_F": None if feature_cols is None else int(len(feature_cols)),
        "label_transform": label_transform,
        "snapshot_dir": SNAP_DIR
    }])
    feat_df = pd.DataFrame({
        "feature_index": list(range(len(feature_cols))) if feature_cols else [],
        "feature_name": list(feature_cols) if feature_cols else []
    })
    return feat_meta, feat_df

def build_week_level_label_stats(paths: List[str], N: int) -> pd.DataFrame:
    """
    Tạo STATS_BY_WEEK (year, epiweek, split, y_real_mean/median/max/zero_rate)
    Không lưu node-level -> nhẹ, đúng paper.
    """
    rows = []
    for p in paths:
        d = _safe_load_pt(p)
        y, w = parse_year_epiweek(p)

        y_log = as_1d_np(d["y"])
        if y_log.size != N:
            raise ValueError(f"{os.path.basename(p)}: y size {y_log.size} != N={N}")
        y_real = np.expm1(y_log)
        y_real = np.clip(y_real, 0.0, None)

        # split tag: do bạn split theo year, mask thường full N
        split = "unknown"
        for s in ["train", "val", "test"]:
            m = d.get(f"{s}_mask", None)
            if m is None:
                continue
            m = as_1d_np(m).astype(bool)
            if m.sum() == N:
                split = s

        rows.append({
            "year": y,
            "epiweek": w,
            "split": split,
            "y_real_mean": float(y_real.mean()),
            "y_real_median": float(np.median(y_real)),
            "y_real_std": float(y_real.std(ddof=1)) if y_real.size > 1 else 0.0,
            "y_real_max": float(y_real.max()),
            "y_zero_rate": float((y_real == 0).mean()),
        })
    df = pd.DataFrame(rows).sort_values(["year", "epiweek"]).reset_index(drop=True)
    return df

def build_year_split_tables(week_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    by_year = week_df.groupby("year").agg(
        weeks=("epiweek", "count"),
        y_real_mean=("y_real_mean", "mean"),
        y_real_median=("y_real_median", "mean"),
        y_real_std=("y_real_std", "mean"),
        y_real_max=("y_real_max", "max"),
        y_zero_rate=("y_zero_rate", "mean"),
    ).reset_index()

    by_split = week_df.groupby("split").agg(
        weeks=("epiweek", "count"),
        y_real_mean=("y_real_mean", "mean"),
        y_real_median=("y_real_median", "mean"),
        y_real_std=("y_real_std", "mean"),
        y_real_max=("y_real_max", "max"),
        y_zero_rate=("y_zero_rate", "mean"),
    ).reset_index()

    return by_year, by_split

def build_overall_label_table(week_df: pd.DataFrame, stats_overall_json: Dict[str, Any] | None) -> pd.DataFrame:
    """
    Tổng quan label: ưu tiên stats_overall_json (nếu có),
    bổ sung thêm quantiles trên week aggregates để trình bày paper.
    """
    rows = []
    if stats_overall_json is not None:
        # đổ key-value ra bảng (paper-friendly)
        for k, v in stats_overall_json.items():
            rows.append({"metric": f"overall_json.{k}", "value": str(v)})

    # quantiles dựa trên per-week median/mean
    week_med = week_df["y_real_median"].to_numpy(dtype=float)
    week_mean = week_df["y_real_mean"].to_numpy(dtype=float)

    for k, v in robust_quantiles(week_med).items():
        rows.append({"metric": f"week_median.{k}", "value": f"{v:.6g}"})
    for k, v in robust_quantiles(week_mean).items():
        rows.append({"metric": f"week_mean.{k}", "value": f"{v:.6g}"})

    rows.append({"metric": "note", "value": "Quantiles computed on per-week aggregates (not all node-week values) to keep report lightweight."})
    return pd.DataFrame(rows)

def make_summary_table(graph_overview_df: pd.DataFrame,
                       feat_meta_df: pd.DataFrame,
                       cov_df: pd.DataFrame,
                       deg_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    1 bảng “Table 4.1” kiểu paper: N/E/F/years/snapshots/degree/isolated nodes
    """
    go = graph_overview_df.iloc[0].to_dict()
    fm = feat_meta_df.iloc[0].to_dict()
    cv = cov_df.iloc[0].to_dict()

    def get_deg(metric: str):
        s = deg_stats_df.loc[deg_stats_df["metric"] == metric, "value"]
        return s.values[0] if len(s) else None

    table = {
        "N_nodes": go["num_nodes_N"],
        "E_edges": go["num_edges_E"],
        "F_features": fm["num_features_F"],
        "years_start": cv["years_start"],
        "years_end": cv["years_end"],
        "snapshots_total": cv["snapshots_total"],
        "deg_mean": get_deg("deg_mean"),
        "deg_p95": get_deg("deg_p95"),
        "deg_max": get_deg("deg_max"),
        "isolated_nodes_deg0": go["isolated_nodes_deg0"],
        "label_transform": fm["label_transform"],
    }
    return pd.DataFrame([table])

# ==============================
# PLOTS (paper figures)
# ==============================
def plot_year_trends(by_year: pd.DataFrame):
    # 1) weeks per year
    plt.figure()
    plt.plot(by_year["year"], by_year["weeks"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Number of weekly snapshots")
    plt.title("Temporal coverage: number of weekly snapshots per year")
    save_fig(os.path.join(FIG_DIR, "fig_4_1_weeks_per_year.png"))

    # 2) y_real_mean trend
    plt.figure()
    plt.plot(by_year["year"], by_year["y_real_mean"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Mean dengue cases per node-week (real scale)")
    plt.title("Label trend by year (mean on real scale)")
    save_fig(os.path.join(FIG_DIR, "fig_4_1_label_mean_by_year.png"))

    # 3) y_real_max by year
    plt.figure()
    plt.plot(by_year["year"], by_year["y_real_max"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Max dengue cases (real scale)")
    plt.title("Extreme weekly cases by year (max on real scale)")
    save_fig(os.path.join(FIG_DIR, "fig_4_1_label_max_by_year.png"))

    # 4) zero rate by year
    plt.figure()
    plt.plot(by_year["year"], by_year["y_zero_rate"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Zero-rate (fraction of nodes with 0 cases)")
    plt.title("Sparsity of dengue counts by year (zero-rate)")
    save_fig(os.path.join(FIG_DIR, "fig_4_1_zero_rate_by_year.png"))

def plot_split_bars(by_split: pd.DataFrame):
    plt.figure()
    plt.bar(by_split["split"], by_split["weeks"])
    plt.xlabel("Split")
    plt.ylabel("Number of weeks")
    plt.title("Temporal split summary (weeks)")
    save_fig(os.path.join(FIG_DIR, "fig_4_1_split_weeks.png"))

    plt.figure()
    plt.bar(by_split["split"], by_split["y_real_mean"])
    plt.xlabel("Split")
    plt.ylabel("Mean dengue cases (real scale)")
    plt.title("Label mean by split (real scale)")
    save_fig(os.path.join(FIG_DIR, "fig_4_1_split_label_mean.png"))

def plot_degree_hist(edge_index: torch.Tensor, N: int):
    src = edge_index[0].cpu().numpy()
    deg = np.bincount(src, minlength=N)

    plt.figure()
    plt.hist(deg, bins=60)
    plt.xlabel("Out-degree")
    plt.ylabel("Count of nodes")
    plt.title("Degree distribution (out-degree) of municipality graph")
    save_fig(os.path.join(FIG_DIR, "fig_4_1_degree_hist.png"))

    # log-scale y for long tail readability
    plt.figure()
    plt.hist(deg, bins=60, log=True)
    plt.xlabel("Out-degree")
    plt.ylabel("Count of nodes (log scale)")
    plt.title("Degree distribution (log scale)")
    save_fig(os.path.join(FIG_DIR, "fig_4_1_degree_hist_log.png"))

def plot_weekly_label_distribution(week_df: pd.DataFrame):
    # distribution of weekly median
    plt.figure()
    plt.hist(week_df["y_real_median"].values, bins=60)
    plt.xlabel("Weekly median cases (real scale)")
    plt.ylabel("Number of weeks")
    plt.title("Distribution of weekly median dengue cases")
    save_fig(os.path.join(FIG_DIR, "fig_4_1_weekly_median_dist.png"))

    # distribution of weekly max (heavy tail)
    plt.figure()
    plt.hist(week_df["y_real_max"].values, bins=60)
    plt.xlabel("Weekly max cases (real scale)")
    plt.ylabel("Number of weeks")
    plt.title("Distribution of weekly maximum dengue cases")
    save_fig(os.path.join(FIG_DIR, "fig_4_1_weekly_max_dist.png"))

# ==============================
# MAIN
# ==============================
def main():
    # load artifacts
    if not os.path.exists(NODE2IDX_PATH):
        raise FileNotFoundError(f"Missing: {NODE2IDX_PATH}")
    with open(NODE2IDX_PATH, "r", encoding="utf-8") as f:
        node2idx = json.load(f)
    N = len(node2idx)

    edge_index = load_edge_index(EDGE_PATH)

    paths = list_snapshots()

    # build tables
    graph_overview_df, deg_stats_df = compute_graph_stats(edge_index, N)
    cov_df, missing_df, year_week_df = compute_time_coverage(paths)

    feat_meta_df, feat_df = build_feature_tables(paths)
    schema_df = scan_snapshot_schema(paths, samples=3)

    stats_overall_json = load_json_if_exists(STATS_OVERALL_JSON)

    # compute label stats from snapshots (reliable + consistent)
    week_df = build_week_level_label_stats(paths, N)
    by_year_df, by_split_df = build_year_split_tables(week_df)
    label_overall_df = build_overall_label_table(week_df, stats_overall_json)

    summary_table_df = make_summary_table(graph_overview_df, feat_meta_df, cov_df, deg_stats_df)
    summary_table_df.to_csv(OUT_TABLE_CSV, index=False, encoding="utf-8")
    print(f"✅ Wrote paper table CSV: {OUT_TABLE_CSV}")

    # optional: also read your precomputed stats csvs (if exist) for cross-checking
    pre_by_year = pd.read_csv(STATS_BY_YEAR_CSV) if os.path.exists(STATS_BY_YEAR_CSV) else None
    pre_by_split = pd.read_csv(STATS_BY_SPLIT_CSV) if os.path.exists(STATS_BY_SPLIT_CSV) else None
    feature_cols_json = load_json_if_exists(FEATURE_COLS_JSON)

    # write Excel (paper bundle)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        summary_table_df.to_excel(writer, sheet_name="TABLE_4_1_SUMMARY", index=False)

        graph_overview_df.to_excel(writer, sheet_name="GRAPH_OVERVIEW", index=False)
        deg_stats_df.to_excel(writer, sheet_name="DEGREE_STATS", index=False)

        cov_df.to_excel(writer, sheet_name="TIME_COVERAGE", index=False)
        missing_df.to_excel(writer, sheet_name="MISSING_WEEKS", index=False)
        year_week_df.to_excel(writer, sheet_name="YEAR_EPIWEEK_LIST", index=False)

        feat_meta_df.to_excel(writer, sheet_name="FEATURES_META", index=False)
        feat_df.to_excel(writer, sheet_name="FEATURES", index=False)

        if os.path.exists(SCALER_PATH):
            scaler = load_json_if_exists(SCALER_PATH)
            # flatten scaler
            rows = []
            def flatten(prefix, obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        flatten(f"{prefix}.{k}" if prefix else k, v)
                else:
                    if isinstance(obj, list) and len(obj) > 100:
                        rows.append({"key": prefix, "value": f"list(len={len(obj)})"})
                    else:
                        rows.append({"key": prefix, "value": str(obj)})
            flatten("", scaler)
            pd.DataFrame(rows).to_excel(writer, sheet_name="SCALER_SUMMARY", index=False)

        schema_df.to_excel(writer, sheet_name="SNAPSHOT_SCHEMA_SAMPLE", index=False)

        label_overall_df.to_excel(writer, sheet_name="LABEL_OVERALL", index=False)
        by_year_df.to_excel(writer, sheet_name="STATS_BY_YEAR_FROM_PT", index=False)
        by_split_df.to_excel(writer, sheet_name="STATS_BY_SPLIT_FROM_PT", index=False)
        week_df.to_excel(writer, sheet_name="STATS_BY_WEEK_FROM_PT", index=False)

        if pre_by_year is not None:
            pre_by_year.to_excel(writer, sheet_name="STATS_BY_YEAR_PRECOMP", index=False)
        if pre_by_split is not None:
            pre_by_split.to_excel(writer, sheet_name="STATS_BY_SPLIT_PRECOMP", index=False)

        if feature_cols_json is not None:
            # store as key/value
            frows = [{"key": k, "value": str(v)} for k, v in feature_cols_json.items()]
            pd.DataFrame(frows).to_excel(writer, sheet_name="FEATURE_COLS_JSON", index=False)

    print(f"✅ Wrote Excel bundle: {OUT_XLSX}")

    # plots for paper section 4.1
    plot_degree_hist(edge_index, N)
    plot_year_trends(by_year_df)
    plot_split_bars(by_split_df)
    plot_weekly_label_distribution(week_df)

    print(f"✅ Wrote figures to: {FIG_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()