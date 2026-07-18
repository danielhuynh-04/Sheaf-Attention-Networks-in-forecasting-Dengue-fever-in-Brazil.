# tools/summarize_dataset_stats.py
import os
import json
import math
import pandas as pd

INPUT_OVERALL = "data/interim/dataset_stats_overall.json"
INPUT_FEATURES = "data/interim/dataset_stats_feature_cols.json"
INPUT_BY_SPLIT = "data/interim/dataset_stats_by_split.csv"
INPUT_BY_YEAR = "data/interim/dataset_stats_by_year.csv"

OUT_DIR = "data/interim"
OUT_JSON = os.path.join(OUT_DIR, "dataset_stats_overall_extended.json")
OUT_CSV  = os.path.join(OUT_DIR, "dataset_stats_overall_extended.csv")
OUT_NOTE = os.path.join(OUT_DIR, "dataset_notes.txt")

def safe_get(d, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    overall = json.load(open(INPUT_OVERALL, "r", encoding="utf-8"))
    feats = json.load(open(INPUT_FEATURES, "r", encoding="utf-8"))
    by_split = pd.read_csv(INPUT_BY_SPLIT)
    by_year = pd.read_csv(INPUT_BY_YEAR)

    # ---- core counts (fallback-friendly) ----
    n_nodes = safe_get(overall, ["n_nodes"], None)
    n_edges = safe_get(overall, ["n_edges"], None)
    n_snapshots = safe_get(overall, ["n_snapshots"], None)

    # ---- graph sparsity ----
    avg_degree = None
    density = None
    if isinstance(n_nodes, (int, float)) and isinstance(n_edges, (int, float)) and n_nodes and n_nodes > 1:
        avg_degree = 2.0 * n_edges / n_nodes
        density = 2.0 * n_edges / (n_nodes * (n_nodes - 1))

    # ---- feature info ----
    feature_cols = feats.get("feature_cols", feats.get("features", []))
    label_col = feats.get("label_col", feats.get("label", "y"))
    label_transform = feats.get("label_transform", "log1p")

    # ---- split coverage ----
    # Expect columns like: split, n_weeks, node_weeks_used, y_real_zero_rate, y_real_p50/p95/p99/max...
    split_summary = by_split.to_dict(orient="records")

    # ---- year coverage flags ----
    # Detect partial year(s): n_weeks < 52
    partial_years = by_year.loc[by_year["n_weeks"] < 52, ["year", "split", "n_weeks"]].to_dict(orient="records")

    # ---- create extended summary ----
    ext = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_snapshots": n_snapshots,
        "avg_degree": avg_degree,
        "graph_density": density,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "label_col": label_col,
        "label_transform": label_transform,
        "split_summary": split_summary,
        "partial_years": partial_years,
    }

    # save json
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(ext, f, ensure_ascii=False, indent=2)

    # save csv (1-row wide format for quick report)
    flat = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_snapshots": n_snapshots,
        "avg_degree": avg_degree,
        "graph_density": density,
        "n_features": len(feature_cols),
        "label_col": label_col,
        "label_transform": label_transform,
        "partial_years_count": len(partial_years),
    }
    pd.DataFrame([flat]).to_csv(OUT_CSV, index=False, encoding="utf-8")

    # write note for paper
    lines = []
    lines.append("Gợi ý mô tả Dataset (đưa vào mục 4.1):")
    lines.append(f"- Đồ thị có N={n_nodes} nút (municipalities), E={n_edges} cạnh không gian.")
    if avg_degree is not None:
        lines.append(f"- Bậc trung bình ~ {avg_degree:.3f}; mật độ đồ thị ~ {density:.6e} (đồ thị rất thưa).")
    lines.append(f"- Số đặc trưng đầu vào F={len(feature_cols)}; nhãn='{label_col}', biến đổi nhãn='{label_transform}'.")
    if partial_years:
        lines.append(f"- Lưu ý các năm chưa đủ 52 tuần (partial year): {partial_years}.")
    lines.append("- Thống kê theo split xem trong dataset_stats_by_split.csv; theo năm xem dataset_stats_by_year.csv.")
    with open(OUT_NOTE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("✅ Wrote:", OUT_JSON)
    print("✅ Wrote:", OUT_CSV)
    print("✅ Wrote:", OUT_NOTE)

if __name__ == "__main__":
    main()