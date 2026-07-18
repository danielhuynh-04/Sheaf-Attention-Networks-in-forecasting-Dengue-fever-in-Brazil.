# tools/export_dataset_excel_report.py
from __future__ import annotations
import os, glob, json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

# ---------------- Paths (edit if needed) ----------------
SNAP_DIR = "data/processed/weekly_pt_scaled"
EDGE_PATH = "data/processed/edge_index.pt"
NODE2IDX_PATH = "data/processed/node2idx.json"
SCALER_PATH = "data/processed/scaler_weekly.json"

OUT_DIR = "data/interim"
OUT_XLSX = os.path.join(OUT_DIR, "dataset_report.xlsx")
OUT_WEEKLY_XLSX = os.path.join(OUT_DIR, "dataset_report_weekly.xlsx")  # optional

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Utils ----------------
def _safe_load_pt(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

def parse_year_epiweek(pt_path: str) -> Tuple[int, int]:
    base = os.path.basename(pt_path).replace(".pt", "")
    y, w = base.split("_")
    return int(y), int(w)

def list_snapshots() -> List[str]:
    paths = sorted(glob.glob(os.path.join(SNAP_DIR, "*.pt")))
    if not paths:
        raise FileNotFoundError(f"No snapshots found in: {SNAP_DIR}")
    return paths

def load_edge_index(path: str) -> torch.Tensor:
    ei = _safe_load_pt(path)
    if isinstance(ei, dict) and "edge_index" in ei:
        ei = ei["edge_index"]
    if not torch.is_tensor(ei):
        raise ValueError("edge_index.pt must be a torch.Tensor (or dict with 'edge_index').")
    if ei.dim() != 2 or ei.size(0) != 2:
        raise ValueError(f"edge_index must be [2, E], got {tuple(ei.shape)}")
    return ei.long()

def as_1d_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.astype(float).reshape(-1)

def quantiles(arr: np.ndarray, qs=(0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)) -> Dict[str, float]:
    out = {}
    for q in qs:
        out[f"q{int(q*100):02d}"] = float(np.quantile(arr, q))
    return out

@dataclass
class RunningStats:
    n: int = 0
    s1: float = 0.0
    s2: float = 0.0
    minv: float = float("inf")
    maxv: float = float("-inf")
    zeros: int = 0

    def update(self, x: np.ndarray):
        if x.size == 0:
            return
        self.n += int(x.size)
        self.s1 += float(x.sum())
        self.s2 += float((x * x).sum())
        self.minv = min(self.minv, float(x.min()))
        self.maxv = max(self.maxv, float(x.max()))
        self.zeros += int((x == 0).sum())

    def mean(self) -> float:
        return float(self.s1 / self.n) if self.n else 0.0

    def std(self) -> float:
        if self.n < 2:
            return 0.0
        var = (self.s2 / self.n) - (self.mean() ** 2)
        return float(max(0.0, var) ** 0.5)

    def zero_rate(self) -> float:
        return float(self.zeros / self.n) if self.n else 0.0

# ---------------- Main export ----------------
def main(write_weekly_xlsx: bool = False, schema_samples: int = 3):
    # Load core artifacts
    with open(NODE2IDX_PATH, "r", encoding="utf-8") as f:
        node2idx = json.load(f)
    N = len(node2idx)

    edge_index = load_edge_index(EDGE_PATH)
    E = int(edge_index.size(1))
    self_loops = int((edge_index[0] == edge_index[1]).sum().item())

    scaler = None
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "r", encoding="utf-8") as f:
            scaler = json.load(f)

    # Degree stats
    src = edge_index[0].numpy()
    deg = np.bincount(src, minlength=N)
    deg_stats = {
        "N_nodes": N,
        "E_edges": E,
        "self_loops": self_loops,
        "deg_min": int(deg.min()),
        "deg_mean": float(deg.mean()),
        "deg_median": float(np.median(deg)),
        "deg_max": int(deg.max()),
        "deg_p95": float(np.quantile(deg, 0.95)),
        "deg_p99": float(np.quantile(deg, 0.99)),
        "isolated_nodes_deg0": int((deg == 0).sum()),
    }

    # Degree histogram table
    # bins chosen for readability in Excel
    bins = np.unique(np.concatenate([np.arange(0, 41, 1), np.array([50, 60, 80, 100, 150, 200, deg.max()+1])]))
    hist, edges = np.histogram(deg, bins=bins)
    degree_hist_df = pd.DataFrame({
        "deg_bin_left": edges[:-1],
        "deg_bin_right": edges[1:],
        "count_nodes": hist
    })

    # Snapshots
    paths = list_snapshots()
    (y0, w0) = parse_year_epiweek(paths[0])
    (y1, w1) = parse_year_epiweek(paths[-1])

    # Global running stats for y_real and y_log
    rs_yreal = RunningStats()
    rs_ylog = RunningStats()

    # Collect per-week rows (760 rows, OK)
    weekly_rows = []

    # For yearly aggregation, we will compute from weekly rows later
    # Feature cols & label_transform from first snapshot
    feature_cols = None
    label_transform = None

    # Split summary tracking (node-level masks)
    split_week_counts = {"train": 0, "val": 0, "test": 0}
    split_nodes_in_split = {"train": [], "val": [], "test": []}

    # Missing week tracking: store (year, epiweek) that appear
    year_week_set = set()

    # Snapshot schema samples
    schema_rows = []

    for idx_p, p in enumerate(paths):
        d = _safe_load_pt(p)
        year, epiweek = parse_year_epiweek(p)
        year_week_set.add((year, epiweek))

        x = d.get("x", None)
        y = d.get("y", None)
        if x is None or y is None:
            raise KeyError(f"{os.path.basename(p)} missing x or y")

        if feature_cols is None and "feature_cols" in d:
            feature_cols = list(d["feature_cols"])
        if label_transform is None and "label_transform" in d:
            label_transform = d["label_transform"]

        # shape checks
        if x.dim() != 2 or x.size(0) != N:
            raise ValueError(f"{os.path.basename(p)}: x must be [N,F] with N={N}, got {tuple(x.shape)}")
        y_np = as_1d_np(y)
        if y_np.size != N:
            raise ValueError(f"{os.path.basename(p)}: y must be [N], got {y_np.size} (expected {N})")

        # y log stats (already log1p if pipeline correct)
        y_log = y_np
        # y real
        y_real = np.expm1(y_log)
        y_real = np.clip(y_real, 0.0, None)

        rs_ylog.update(y_log)
        rs_yreal.update(y_real)

        # per-week stats
        wk = {
            "year": year,
            "epiweek": epiweek,
            "N_nodes": N,
            "F_features": int(x.size(1)),
            "y_real_mean": float(y_real.mean()),
            "y_real_median": float(np.median(y_real)),
            "y_real_std": float(y_real.std(ddof=1)) if y_real.size > 1 else 0.0,
            "y_real_max": float(y_real.max()),
            "y_zero_rate": float((y_real == 0).mean()),
        }

        # Determine split type for the week (based on which mask is non-empty)
        # Your pipeline splits by year, so one of these is typically full True.
        split_tag = "unknown"
        for split in ["train", "val", "test"]:
            m = d.get(f"{split}_mask", None)
            if m is None:
                continue
            m = as_1d_np(m).astype(bool)
            if m.sum() > 0:
                # record nodes_in_split distribution
                split_nodes_in_split[split].append(int(m.sum()))
                split_week_counts[split] += 1
                # pick the dominant split for the week
                if m.sum() == N:
                    split_tag = split
        wk["split"] = split_tag

        weekly_rows.append(wk)

        # Schema sample rows (first few)
        if idx_p < schema_samples:
            keys = sorted(list(d.keys()))
            schema_rows.append({
                "sample_file": os.path.basename(p),
                "keys": ", ".join(keys),
                "x_shape": str(tuple(x.shape)),
                "y_shape": str(tuple(np.asarray(y_np).shape)),
                "has_geocodes": "geocodes" in d,
                "has_feature_cols": "feature_cols" in d,
                "has_scaler": "scaler" in d,
                "has_masks": all((k in d) for k in ["train_mask", "val_mask", "test_mask"]) or
                             any((k in d) for k in ["train_mask", "val_mask", "test_mask"]),
                "label_transform": d.get("label_transform", None),
            })

    weekly_df = pd.DataFrame(weekly_rows).sort_values(["year", "epiweek"]).reset_index(drop=True)

    # Yearly summary derived from weekly_df
    by_year_df = weekly_df.groupby("year").agg(
        weeks=("epiweek", "count"),
        y_real_mean=("y_real_mean", "mean"),
        y_real_median=("y_real_median", "mean"),
        y_real_std=("y_real_std", "mean"),
        y_real_max=("y_real_max", "max"),
        y_zero_rate=("y_zero_rate", "mean"),
    ).reset_index()

    # Split summary
    split_rows = []
    for split in ["train", "val", "test"]:
        if split_week_counts[split] == 0:
            continue
        arr = np.array(split_nodes_in_split[split], dtype=float) if split_nodes_in_split[split] else np.array([], dtype=float)
        split_rows.append({
            "split": split,
            "weeks": int(split_week_counts[split]),
            "nodes_total_per_week": int(N),
            "nodes_in_split_mean": float(arr.mean()) if arr.size else 0.0,
            "nodes_in_split_min": int(arr.min()) if arr.size else 0,
            "nodes_in_split_max": int(arr.max()) if arr.size else 0,
        })
    by_split_df = pd.DataFrame(split_rows)

    # Missing week detection (basic): within each year, expected epiweek range min..max present
    missing_rows = []
    for y in sorted(weekly_df["year"].unique()):
        weeks_present = sorted(weekly_df.loc[weekly_df["year"] == y, "epiweek"].tolist())
        if not weeks_present:
            continue
        wmin, wmax = min(weeks_present), max(weeks_present)
        expected = set(range(int(wmin), int(wmax) + 1))
        missing = sorted(list(expected - set(weeks_present)))
        for mw in missing:
            missing_rows.append({"year": int(y), "missing_epiweek": int(mw)})
    missing_df = pd.DataFrame(missing_rows)

    # Overall label stats (quantiles computed from weekly aggregate is NOT enough)
    # For Excel, we report reliable running mean/std/min/max/zero_rate,
    # and ALSO compute quantiles using a subsample from weekly_df by taking max/median etc.
    # If you want true global quantiles over all node-week values, you'd need saving y_real_all (heavy).
    # Here we compute quantiles over per-week medians (paper-friendly).
    yweek_median = weekly_df["y_real_median"].to_numpy(dtype=float)
    yweek_mean = weekly_df["y_real_mean"].to_numpy(dtype=float)
    label_overall = {
        "y_log_mean": rs_ylog.mean(),
        "y_log_std": rs_ylog.std(),
        "y_log_min": rs_ylog.minv,
        "y_log_max": rs_ylog.maxv,
        "y_real_mean": rs_yreal.mean(),
        "y_real_std": rs_yreal.std(),
        "y_real_min": rs_yreal.minv,
        "y_real_max": rs_yreal.maxv,
        "y_real_zero_rate": rs_yreal.zero_rate(),
        "note_quantiles": "Quantiles below are computed over per-week aggregates (median/mean), not all node-week values.",
        **{f"week_median_{k}": v for k, v in quantiles(yweek_median).items()},
        **{f"week_mean_{k}": v for k, v in quantiles(yweek_mean).items()},
    }
    label_overall_df = pd.DataFrame(list(label_overall.items()), columns=["metric", "value"])

    # Features sheet
    feat_df = pd.DataFrame({
        "feature_index": list(range(len(feature_cols))) if feature_cols else [],
        "feature_name": feature_cols if feature_cols else []
    })
    feat_meta_df = pd.DataFrame([
        {"num_features": None if feature_cols is None else len(feature_cols),
         "label_transform": label_transform,
         "scaler_path": SCALER_PATH if os.path.exists(SCALER_PATH) else None,
         "snap_dir": SNAP_DIR}
    ])

    # Scaler summary (best-effort; structure varies)
    scaler_df = pd.DataFrame()
    if scaler is not None:
        # dump top-level keys for readability
        rows = []
        def flatten(prefix, obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    flatten(f"{prefix}.{k}" if prefix else k, v)
            else:
                # avoid huge arrays
                if isinstance(obj, list) and len(obj) > 50:
                    rows.append({"key": prefix, "value": f"list(len={len(obj)})"})
                else:
                    rows.append({"key": prefix, "value": str(obj)})
        flatten("", scaler)
        scaler_df = pd.DataFrame(rows)

    # Graph overview sheet
    graph_overview_df = pd.DataFrame([
        {"num_nodes": N, "num_edges": E, "self_loops": self_loops,
         "snapshots_total": len(paths), "time_start": f"{y0}_{w0}", "time_end": f"{y1}_{w1}"}
    ])
    degree_stats_df = pd.DataFrame(list(deg_stats.items()), columns=["metric", "value"])

    # Time coverage sheet
    weeks_per_year_df = by_year_df[["year", "weeks"]].copy()
    time_cov_df = pd.DataFrame([
        {"time_start": f"{y0}_{w0}", "time_end": f"{y1}_{w1}", "snapshots_total": len(paths),
         "years_covered": int(by_year_df["year"].nunique()), "notes": "Missing weeks (if any) listed in sheet MISSING_WEEKS."}
    ])

    # Snapshot schema sample
    schema_df = pd.DataFrame(schema_rows)

    # README sheet (for your paper writing)
    readme_lines = [
        "Dataset Report (Excel) - purpose: show dataset completeness and integrity for paper Section 4.1",
        f"Snapshots: {SNAP_DIR}",
        f"Graph: {EDGE_PATH}",
        f"Node mapping: {NODE2IDX_PATH}",
        f"Scaler: {SCALER_PATH}",
        "",
        "Key sheets:",
        "- GRAPH_OVERVIEW / DEGREE_STATS / DEGREE_HIST",
        "- TIME_COVERAGE / MISSING_WEEKS",
        "- FEATURES / SCALER_SUMMARY",
        "- LABEL_OVERALL / STATS_BY_YEAR / STATS_BY_SPLIT / STATS_BY_WEEK",
        "- SNAPSHOT_SCHEMA_SAMPLE",
    ]
    readme_df = pd.DataFrame({"text": readme_lines})

    # ---------------- Write Excel ----------------
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        readme_df.to_excel(writer, sheet_name="README", index=False)
        graph_overview_df.to_excel(writer, sheet_name="GRAPH_OVERVIEW", index=False)
        degree_stats_df.to_excel(writer, sheet_name="DEGREE_STATS", index=False)
        degree_hist_df.to_excel(writer, sheet_name="DEGREE_HIST", index=False)

        time_cov_df.to_excel(writer, sheet_name="TIME_COVERAGE", index=False)
        weeks_per_year_df.to_excel(writer, sheet_name="WEEKS_PER_YEAR", index=False)
        missing_df.to_excel(writer, sheet_name="MISSING_WEEKS", index=False)

        feat_meta_df.to_excel(writer, sheet_name="FEATURES_META", index=False)
        feat_df.to_excel(writer, sheet_name="FEATURES", index=False)

        if not scaler_df.empty:
            scaler_df.to_excel(writer, sheet_name="SCALER_SUMMARY", index=False)

        label_overall_df.to_excel(writer, sheet_name="LABEL_OVERALL", index=False)
        by_year_df.to_excel(writer, sheet_name="STATS_BY_YEAR", index=False)
        by_split_df.to_excel(writer, sheet_name="STATS_BY_SPLIT", index=False)

        # weekly may be big-ish; put into main file but you can disable if you want
        weekly_df.to_excel(writer, sheet_name="STATS_BY_WEEK", index=False)

        schema_df.to_excel(writer, sheet_name="SNAPSHOT_SCHEMA_SAMPLE", index=False)

    print(f"✅ Wrote Excel: {OUT_XLSX}")

    # Optional: separate weekly file
    if write_weekly_xlsx:
        with pd.ExcelWriter(OUT_WEEKLY_XLSX, engine="openpyxl") as writer:
            weekly_df.to_excel(writer, sheet_name="STATS_BY_WEEK", index=False)
            print(f"✅ Wrote Weekly-only Excel: {OUT_WEEKLY_XLSX}")

if __name__ == "__main__":
    # set True if you want a separate weekly-only xlsx
    main(write_weekly_xlsx=False, schema_samples=3)