# tools/dataset_stats_from_raw_weekly.py
from __future__ import annotations
import os, glob
import numpy as np
import pandas as pd

# ====== CONFIG: chỉnh đúng thư mục raw weekly theo năm của bạn ======
RAW_YEARLY_DIR = "data/interim/yearly"   # ví dụ: data/interim/yearly/weekly_2010.csv
PATTERN = "weekly_*.csv"
YEAR_MIN, YEAR_MAX = 2010, 2024

OUT_DIR = "data/interim/dataset_raw_stats"
OUT_XLSX = os.path.join(OUT_DIR, "dataset_raw_weekly_stats_2010_2024.xlsx")
OUT_BY_YEAR_CSV = os.path.join(OUT_DIR, "raw_stats_by_year.csv")
OUT_OVERALL_CSV = os.path.join(OUT_DIR, "raw_stats_overall.csv")
OUT_MISSING_KEYS_CSV = os.path.join(OUT_DIR, "raw_missing_keys_by_year.csv")

os.makedirs(OUT_DIR, exist_ok=True)


def _safe_read_csv(path: str) -> pd.DataFrame:
    # cố gắng đọc nhanh & ổn định
    return pd.read_csv(path, low_memory=False)


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    # chuẩn hoá tên cột hay gặp
    rename = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ["municipality", "muni", "munic", "munic_res"] and "geocode" not in df.columns:
            rename[c] = "geocode"
        if lc in ["week", "epi_week"] and "epiweek" not in df.columns:
            rename[c] = "epiweek"
        if lc in ["cases", "case", "casos"] and "casos" not in df.columns:
            rename[c] = "casos"
    if rename:
        df = df.rename(columns=rename)

    # ép kiểu tối thiểu
    if "geocode" in df.columns:
        df["geocode"] = df["geocode"].astype(str).str.strip()
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    if "epiweek" in df.columns:
        df["epiweek"] = pd.to_numeric(df["epiweek"], errors="coerce")
    if "casos" in df.columns:
        df["casos"] = pd.to_numeric(df["casos"], errors="coerce")
    return df


def _quantiles(s: pd.Series, qs=(0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {f"q{int(q*100):02d}": np.nan for q in qs}
    return {f"q{int(q*100):02d}": float(s.quantile(q)) for q in qs}


def per_year_stats(df: pd.DataFrame, year: int) -> tuple[dict, pd.DataFrame]:
    # bắt buộc tối thiểu: geocode, epiweek, casos
    required = ["geocode", "epiweek", "casos"]
    missing_required = [c for c in required if c not in df.columns]

    n_rows = len(df)
    n_geo = df["geocode"].nunique() if "geocode" in df.columns else np.nan
    n_weeks = df["epiweek"].nunique() if "epiweek" in df.columns else np.nan

    # độ phân giải geocode×epiweek
    res_expected = (n_geo * n_weeks) if (pd.notna(n_geo) and pd.notna(n_weeks)) else np.nan
    res_actual = df.dropna(subset=["geocode", "epiweek"]).drop_duplicates(["geocode", "epiweek"]).shape[0] if ("geocode" in df.columns and "epiweek" in df.columns) else np.nan
    coverage = (res_actual / res_expected) if (isinstance(res_expected, (int, float)) and res_expected and pd.notna(res_actual)) else np.nan

    # label stats
    casos = df["casos"] if "casos" in df.columns else pd.Series(dtype=float)
    casos_num = pd.to_numeric(casos, errors="coerce")
    zero_rate = float((casos_num.fillna(0) == 0).mean()) if n_rows else np.nan

    label_stats = {
        "casos_mean": float(casos_num.mean()) if n_rows else np.nan,
        "casos_median": float(casos_num.median()) if n_rows else np.nan,
        "casos_std": float(casos_num.std(ddof=1)) if n_rows > 1 else np.nan,
        "casos_max": float(casos_num.max()) if n_rows else np.nan,
        "casos_zero_rate": zero_rate
    }
    label_stats.update({f"casos_{k}": v for k, v in _quantiles(casos_num).items()})

    # missing rate theo feature
    miss_rates = {}
    for c in df.columns:
        miss_rates[f"miss_{c}"] = float(df[c].isna().mean())

    # kiểm tra thiếu khóa geocode×epiweek (nếu muốn “độ phủ”)
    missing_keys_df = pd.DataFrame()
    if ("geocode" in df.columns) and ("epiweek" in df.columns):
        weeks = sorted(df["epiweek"].dropna().astype(int).unique().tolist())
        geos = sorted(df["geocode"].dropna().astype(str).unique().tolist())
        present = set(zip(df["geocode"].astype(str), df["epiweek"].astype(int)))
        # Chỉ liệt kê thiếu theo tuần (nhẹ hơn): mỗi tuần thiếu bao nhiêu geocode
        rows = []
        for w in weeks:
            cnt_present = sum(((g, w) in present) for g in geos)
            rows.append({
                "year": year,
                "epiweek": int(w),
                "geocodes_present": int(cnt_present),
                "geocodes_total": int(len(geos)),
                "missing_geocodes": int(len(geos) - cnt_present),
                "coverage_week": float(cnt_present / max(1, len(geos)))
            })
        missing_keys_df = pd.DataFrame(rows)

    out = {
        "year": year,
        "rows": int(n_rows),
        "unique_geocodes": int(n_geo) if pd.notna(n_geo) else np.nan,
        "unique_epiweeks": int(n_weeks) if pd.notna(n_weeks) else np.nan,
        "resolution_expected_geo_x_week": float(res_expected) if pd.notna(res_expected) else np.nan,
        "resolution_actual_pairs": float(res_actual) if pd.notna(res_actual) else np.nan,
        "coverage_ratio": float(coverage) if pd.notna(coverage) else np.nan,
        "missing_required_cols": ", ".join(missing_required) if missing_required else ""
    }
    out.update(label_stats)
    out.update(miss_rates)
    return out, missing_keys_df


def main():
    files = sorted(glob.glob(os.path.join(RAW_YEARLY_DIR, PATTERN)))
    # lọc đúng 2010..2024
    year_files = []
    for f in files:
        base = os.path.basename(f).replace(".csv", "")
        # weekly_2010
        try:
            y = int(base.split("_")[-1])
        except Exception:
            continue
        if YEAR_MIN <= y <= YEAR_MAX:
            year_files.append((y, f))
    year_files = sorted(year_files, key=lambda x: x[0])

    if not year_files:
        raise FileNotFoundError(f"Không tìm thấy weekly_{YEAR_MIN}..weekly_{YEAR_MAX}.csv trong {RAW_YEARLY_DIR}")

    rows_year = []
    missing_week_rows = []
    all_df_list = []

    for y, path in year_files:
        df = _safe_read_csv(path)
        df = _ensure_cols(df)
        # nếu file không có year, bổ sung year từ filename
        if "year" not in df.columns:
            df["year"] = y

        stat, miss_week_df = per_year_stats(df, y)
        rows_year.append(stat)

        if not miss_week_df.empty:
            missing_week_rows.append(miss_week_df)

        # giữ lại để thống kê overall (nhưng hạn chế cột nặng nếu file quá lớn)
        keep_cols = [c for c in df.columns if c in ["geocode", "year", "epiweek", "casos"] or c.lower().endswith("_med") or c.lower() in ["precip_tot", "rainy_days", "temp_med", "rel_humid_med"]]
        if keep_cols:
            all_df_list.append(df[keep_cols].copy())
        else:
            all_df_list.append(df[["geocode", "year", "epiweek", "casos"]].copy())

        print(f"Loaded {os.path.basename(path)} | rows={len(df):,} | geocodes={df['geocode'].nunique() if 'geocode' in df.columns else 'NA'}")

    by_year = pd.DataFrame(rows_year)

    # overall
    big = pd.concat(all_df_list, ignore_index=True)
    big = _ensure_cols(big)

    overall = {
        "years_covered": f"{YEAR_MIN}-{YEAR_MAX}",
        "total_rows": int(len(big)),
        "unique_geocodes_total": int(big["geocode"].nunique()) if "geocode" in big.columns else np.nan,
        "unique_epiweeks_total": int(big["epiweek"].nunique()) if "epiweek" in big.columns else np.nan,
        "unique_years_total": int(big["year"].nunique()) if "year" in big.columns else np.nan,
    }

    if ("geocode" in big.columns) and ("year" in big.columns) and ("epiweek" in big.columns):
        overall["unique_geocode_year_week"] = int(big.dropna(subset=["geocode","year","epiweek"]).drop_duplicates(["geocode","year","epiweek"]).shape[0])

    casos_num = pd.to_numeric(big["casos"], errors="coerce") if "casos" in big.columns else pd.Series(dtype=float)
    overall.update({
        "casos_mean": float(casos_num.mean()),
        "casos_median": float(casos_num.median()),
        "casos_std": float(casos_num.std(ddof=1)) if len(casos_num.dropna()) > 1 else np.nan,
        "casos_max": float(casos_num.max()),
        "casos_zero_rate": float((casos_num.fillna(0) == 0).mean())
    })
    overall.update({f"casos_{k}": v for k, v in _quantiles(casos_num).items()})

    overall_df = pd.DataFrame([overall])

    # missing by week
    missing_by_week = pd.concat(missing_week_rows, ignore_index=True) if missing_week_rows else pd.DataFrame()

    # save
    by_year.to_csv(OUT_BY_YEAR_CSV, index=False, encoding="utf-8")
    overall_df.to_csv(OUT_OVERALL_CSV, index=False, encoding="utf-8")
    if not missing_by_week.empty:
        missing_by_week.to_csv(OUT_MISSING_KEYS_CSV, index=False, encoding="utf-8")

    # Excel bundle
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        overall_df.to_excel(writer, sheet_name="OVERALL", index=False)
        by_year.to_excel(writer, sheet_name="BY_YEAR", index=False)
        if not missing_by_week.empty:
            missing_by_week.to_excel(writer, sheet_name="MISSING_BY_WEEK", index=False)

        # thêm sheet: top years theo max cases
        top_max = by_year[["year","casos_max","casos_mean","casos_zero_rate","rows","unique_geocodes","unique_epiweeks"]].sort_values("casos_max", ascending=False)
        top_max.to_excel(writer, sheet_name="TOP_EXTREMES_BY_YEAR", index=False)

    print(f"\n✅ Saved:")
    print(f" - {OUT_BY_YEAR_CSV}")
    print(f" - {OUT_OVERALL_CSV}")
    if not missing_by_week.empty:
        print(f" - {OUT_MISSING_KEYS_CSV}")
    print(f" - {OUT_XLSX}")


if __name__ == "__main__":
    main()