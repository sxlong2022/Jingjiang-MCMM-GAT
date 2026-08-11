import argparse
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Paths:
    codes_dir: Path
    q_qs_shashi_csv: Path
    d50_measured_csv: Path
    d50_forecast_csv: Path
    out_dir: Path


_MONTH_MAP = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def _read_csv_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "cp936", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception:
            continue
    raw = path.read_bytes()
    text = raw.decode("gb18030", errors="ignore")
    return pd.read_csv(io.StringIO(text), **kwargs)


def _parse_yyyymm_to_timestamp(yyyymm: str) -> pd.Timestamp:
    yyyymm = str(yyyymm).strip()
    if len(yyyymm) != 6 or not yyyymm.isdigit():
        raise ValueError(f"Invalid yyyymm: {yyyymm}")
    y = int(yyyymm[:4])
    m = int(yyyymm[4:6])
    return pd.Timestamp(year=y, month=m, day=1)


def _format_yyyymm(ts: pd.Timestamp) -> str:
    return f"{ts.year:04d}{ts.month:02d}"


def load_q_qs_shashi_1991_2024(path: Path) -> pd.DataFrame:
    """Read Shashi monthly discharge and sediment load (1991–2024).

    Note: The original file header may be GBK/CP936 encoded with garbled characters.
    We read as 'headerless three columns' and filter out the header row via numeric conversion.

    Output columns: date, yyyymm, Q_m3s, Qs
    """
    df = _read_csv_with_fallback(path, header=None)
    if df.shape[1] < 3:
        raise ValueError(f"Invalid Q csv format (need >=3 columns): {path}")

    df = df.iloc[:, :4] if df.shape[1] >= 4 else df.iloc[:, :3]
    if df.shape[1] >= 4:
        df.columns = ["Year", "Month", "Q", "Qs"]
    else:
        df.columns = ["Year", "Month", "Q"]

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    df["Q"] = pd.to_numeric(df["Q"], errors="coerce")
    if "Qs" in df.columns:
        df["Qs"] = pd.to_numeric(df["Qs"], errors="coerce")

    df = df.dropna(subset=["Year", "Month", "Q"]).copy()
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)

    df = df[(df["Month"] >= 1) & (df["Month"] <= 12)].copy()

    df["date"] = pd.to_datetime(
        df[["Year", "Month"]]
        .rename(columns={"Year": "year", "Month": "month"})
        .assign(day=1)
    )

    df["yyyymm"] = df["date"].dt.strftime("%Y%m")
    df = df.sort_values("date").drop_duplicates(subset=["yyyymm"], keep="last")

    out_cols = ["date", "yyyymm", "Q"]
    if "Qs" in df.columns:
        out_cols.append("Qs")
    out = df[out_cols].rename(columns={"Q": "Q_m3s"}).reset_index(drop=True)
    return out

def _parse_d50_short_date(s: str) -> pd.Timestamp:
    parts = str(s).strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid D50 Date format: {s}")
    yy = int(parts[0])
    mm = _MONTH_MAP[parts[1]]
    return pd.Timestamp(year=2000 + yy, month=mm, day=1)


def load_d50_series_m(path_measured: Path, path_forecast: Path) -> pd.DataFrame:
    """Merge D50 (monthly) measured (2009-2014) and Prophet forecast (2015-2030).

    Output columns: date, yyyymm, D50_m, D50_std_m, d50_source
    - D50 units unified to meter (m)
    """
    df_m = _read_csv_with_fallback(path_measured)
    if not {"Date", "D50"}.issubset(df_m.columns):
        raise ValueError(f"Invalid measured D50 csv columns: {path_measured}")
    df_m = df_m.copy()
    df_m["date"] = df_m["Date"].apply(_parse_d50_short_date)
    df_m["D50_m"] = pd.to_numeric(df_m["D50"], errors="coerce") / 1000.0  # mm -> m
    df_m["D50_std_m"] = np.nan
    df_m["d50_source"] = "measured"

    df_f = _read_csv_with_fallback(path_forecast)
    if not {"Date", "Forecast Mean", "Forecast Std Dev"}.issubset(df_f.columns):
        raise ValueError(f"Invalid forecast D50 csv columns: {path_forecast}")
    df_f = df_f.copy()
    df_f["date"] = pd.to_datetime(df_f["Date"], errors="coerce")
    df_f["D50_m"] = pd.to_numeric(df_f["Forecast Mean"], errors="coerce") / 1000.0
    df_f["D50_std_m"] = pd.to_numeric(df_f["Forecast Std Dev"], errors="coerce") / 1000.0
    df_f["d50_source"] = "prophet_forecast"

    df = pd.concat(
        [
            df_m[["date", "D50_m", "D50_std_m", "d50_source"]],
            df_f[["date", "D50_m", "D50_std_m", "d50_source"]],
        ],
        axis=0,
        ignore_index=True,
    )

    df = df.dropna(subset=["date", "D50_m"]).copy()
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp(how="start")
    df["yyyymm"] = df["date"].dt.strftime("%Y%m")
    df = df.sort_values("date").drop_duplicates(subset=["yyyymm"], keep="last")

    return df[["date", "yyyymm", "D50_m", "D50_std_m", "d50_source"]].reset_index(drop=True)


def build_future_q_qs_ensemble(
    q_qs_hist: pd.DataFrame,
    start_yyyymm: str,
    end_yyyymm: str,
    n_members: int,
    seed: int,
    method: str,
    scale: float,
) -> pd.DataFrame:
    """Generate future Q/Qs scenarios (2025–2030).

    method:
    - bootstrap_year: Year-block bootstrap sampling (preserves seasonal pattern and intra-annual correlation)
    - climatology: Use historical monthly means (deterministic)

    Output columns: member, date, yyyymm, Q_m3s, Qs, q_method
    """
    start = _parse_yyyymm_to_timestamp(start_yyyymm)
    end = _parse_yyyymm_to_timestamp(end_yyyymm)
    months = pd.date_range(start=start, end=end, freq="MS")

    q_qs_hist = q_qs_hist.copy()
    q_qs_hist["year"] = q_qs_hist["date"].dt.year
    q_qs_hist["month"] = q_qs_hist["date"].dt.month

    years_available = sorted(q_qs_hist["year"].unique().tolist())
    # Force sampling only from years >= 2003 to prevent pre-dam high-sediment years (1991-2002) from leaking into future
    years_available = [y for y in years_available if y >= 2003]
    if not years_available:
        raise ValueError("No historical Q data >= 2003")

    clim_q = q_qs_hist.groupby("month")["Q_m3s"].mean().to_dict()
    has_qs = "Qs" in q_qs_hist.columns
    clim_qs = q_qs_hist.groupby("month")["Qs"].mean().to_dict() if has_qs else {}

    rng = np.random.default_rng(int(seed))
    rows = []

    if method not in {"bootstrap_year", "climatology"}:
        raise ValueError(f"Unknown q method: {method}")

    q_lookup = {(int(r.year), int(r.month)): float(r.Q_m3s) for r in q_qs_hist.itertuples(index=False)}
    qs_lookup = None
    if has_qs:
        qs_lookup = {(int(r.year), int(r.month)): float(r.Qs) for r in q_qs_hist.itertuples(index=False)}

    n_years_future = len(sorted({(d.year) for d in months}))

    for member in range(n_members):
        if method == "bootstrap_year":
            sampled_years = rng.choice(years_available, size=n_years_future, replace=True)
            year_map = {int(y): int(sampled_years[i]) for i, y in enumerate(sorted({d.year for d in months}))}

            for dt in months:
                src_y = year_map[int(dt.year)]
                src_key = (int(src_y), int(dt.month))
                q = q_lookup.get(src_key, np.nan)
                if not np.isfinite(q):
                    q = float(clim_q.get(int(dt.month), np.nan))

                if qs_lookup is not None:
                    qs = qs_lookup.get(src_key, np.nan)
                    if not np.isfinite(qs):
                        qs = float(clim_qs.get(int(dt.month), np.nan))
                else:
                    qs = np.nan

                q = float(q) * float(scale)
                if np.isfinite(qs):
                    qs = float(qs) * float(scale)

                rows.append(
                    {
                        "member": int(member),
                        "date": dt,
                        "yyyymm": _format_yyyymm(dt),
                        "Q_m3s": q,
                        "Qs": qs,
                        "q_method": f"bootstrap_year(scale={scale})",
                        "q_src_year": int(src_y),
                    }
                )
        else:
            for dt in months:
                q = float(clim_q.get(int(dt.month), np.nan)) * float(scale)
                qs = float(clim_qs.get(int(dt.month), np.nan)) * float(scale) if clim_qs else np.nan
                rows.append(
                    {
                        "member": int(member),
                        "date": dt,
                        "yyyymm": _format_yyyymm(dt),
                        "Q_m3s": q,
                        "Qs": qs,
                        "q_method": f"climatology(scale={scale})",
                        "q_src_year": np.nan,
                    }
                )

    return pd.DataFrame(rows)


def build_future_d50_ensemble(
    d50_all: pd.DataFrame,
    start_yyyymm: str,
    end_yyyymm: str,
    n_members: int,
    seed: int,
    clip_m: tuple[float, float] = (0.0001, 0.0006),
) -> pd.DataFrame:
    """Future D50 ensemble: Gaussian perturbation sampling using Prophet mean/std, clipped to physical range."""
    start = _parse_yyyymm_to_timestamp(start_yyyymm)
    end = _parse_yyyymm_to_timestamp(end_yyyymm)
    months = pd.date_range(start=start, end=end, freq="MS")

    d50_all = d50_all.copy()
    d50_lookup = d50_all.set_index("yyyymm")[["D50_m", "D50_std_m", "d50_source"]]

    rng = np.random.default_rng(int(seed) + 999)

    rows = []
    lo, hi = float(clip_m[0]), float(clip_m[1])

    for member in range(n_members):
        z = rng.standard_normal(size=len(months))
        for i, dt in enumerate(months):
            ym = _format_yyyymm(dt)
            if ym not in d50_lookup.index:
                mean = np.nan
                std = np.nan
                src = "missing"
            else:
                rec = d50_lookup.loc[ym]
                mean = float(rec["D50_m"]) if np.isfinite(rec["D50_m"]) else np.nan
                std = float(rec["D50_std_m"]) if np.isfinite(rec["D50_std_m"]) else 0.0
                src = str(rec["d50_source"])

            if not np.isfinite(mean):
                sample = np.nan
            else:
                sample = mean + float(z[i]) * float(std)
                sample = float(np.clip(sample, lo, hi))

            rows.append(
                {
                    "member": int(member),
                    "date": dt,
                    "yyyymm": ym,
                    "D50_m": mean,
                    "D50_std_m": std,
                    "D50_sample_m": sample,
                    "d50_source": src,
                }
            )

    return pd.DataFrame(rows)


def make_q_qs_historical(q_qs: pd.DataFrame) -> pd.DataFrame:
    cols = ["date", "yyyymm", "Q_m3s"]
    if "Qs" in q_qs.columns:
        cols.append("Qs")
    else:
        q_qs = q_qs.copy()
        q_qs["Qs"] = np.nan
        cols.append("Qs")
    return q_qs[cols].sort_values("date").reset_index(drop=True)


def qc_month_coverage(df: pd.DataFrame, yyyymm_col: str, start_yyyymm: str, end_yyyymm: str) -> pd.DataFrame:
    start = _parse_yyyymm_to_timestamp(start_yyyymm)
    end = _parse_yyyymm_to_timestamp(end_yyyymm)
    expected = pd.date_range(start=start, end=end, freq="MS")
    expected_ym = {_format_yyyymm(d) for d in expected}

    got = set(df[yyyymm_col].astype(str).tolist())
    missing = sorted(expected_ym - got)

    return pd.DataFrame({"missing_yyyymm": missing})


def main():
    parser = argparse.ArgumentParser(description="Build Jingjiang forcing (Q + D50) scenarios for 2025-2030")

    parser.add_argument("--start", default="202501", help="Start month yyyymm")
    parser.add_argument("--end", default="203012", help="End month yyyymm")
    parser.add_argument("--n_members", type=int, default=30, help="Ensemble size per scenario")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--q_method",
        default="bootstrap_year",
        choices=["bootstrap_year", "climatology"],
        help="How to generate future discharge Q",
    )

    parser.add_argument(
        "--scenarios",
        default="S0,Swet,Sdry",
        help="Comma-separated scenario ids",
    )

    parser.add_argument("--scale_S0", type=float, default=1.0)
    parser.add_argument("--scale_Swet", type=float, default=1.15)
    parser.add_argument("--scale_Sdry", type=float, default=0.85)

    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: data/processed)",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_raw_dir = repo_root / "data" / "raw"

    paths = Paths(
        codes_dir=repo_root,
        q_qs_shashi_csv=data_raw_dir / "shashi_monthly_q_qs.csv",
        d50_measured_csv=data_raw_dir / "d50_measured.csv",
        d50_forecast_csv=data_raw_dir / "d50_prophet_forecast.csv",
        out_dir=Path(args.out_dir) if args.out_dir else (repo_root / "data" / "processed"),
    )

    paths.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load historical Q + Qs
    q_qs_shashi = load_q_qs_shashi_1991_2024(paths.q_qs_shashi_csv)
    q_qs_shashi.to_csv(paths.out_dir / "q_qs_shashi_1991_2024_clean.csv", index=False)

    forcing_hist = make_q_qs_historical(q_qs_shashi)
    forcing_hist.to_csv(paths.out_dir / "forcing_q_qs_historical_1991_2024.csv", index=False)

    # 3) Load D50 (measured + forecast)
    d50_all = load_d50_series_m(paths.d50_measured_csv, paths.d50_forecast_csv)
    d50_all.to_csv(paths.out_dir / "d50_shashi_2009_2030_m.csv", index=False)

    # 4) Future ensembles
    scenarios = [s.strip() for s in str(args.scenarios).split(",") if s.strip()]
    scale_map = {
        "S0": float(args.scale_S0),
        "Swet": float(args.scale_Swet),
        "Sdry": float(args.scale_Sdry),
    }

    all_rows = []
    for sc in scenarios:
        scale = float(scale_map.get(sc, 1.0))
        q_future = build_future_q_qs_ensemble(
            q_qs_hist=forcing_hist,
            start_yyyymm=args.start,
            end_yyyymm=args.end,
            n_members=int(args.n_members),
            seed=int(args.seed) + (abs(hash(sc)) % 100000),
            method=str(args.q_method),
            scale=scale,
        )
        q_future["scenario"] = sc
        all_rows.append(q_future)

    q_future_all = pd.concat(all_rows, axis=0, ignore_index=True)

    d50_future = build_future_d50_ensemble(
        d50_all=d50_all,
        start_yyyymm=args.start,
        end_yyyymm=args.end,
        n_members=int(args.n_members),
        seed=int(args.seed),
    )

    out = q_future_all.merge(d50_future, on=["member", "date", "yyyymm"], how="left")

    out = out[
        [
            "scenario",
            "member",
            "date",
            "yyyymm",
            "Q_m3s",
            "Qs",
            "q_method",
            "q_src_year",
            "D50_m",
            "D50_std_m",
            "D50_sample_m",
            "d50_source",
        ]
    ].sort_values(["scenario", "member", "date"])

    out.to_csv(paths.out_dir / f"forcing_q_d50_scenarios_{args.start}_{args.end}.csv", index=False)

    # 5) QC
    qc_q = qc_month_coverage(q_qs_shashi, "yyyymm", "199101", "202412")
    qc_q.to_csv(paths.out_dir / "qc_missing_q_1991_2024.csv", index=False)

    qc_d50 = qc_month_coverage(d50_all, "yyyymm", "200901", "203012")
    qc_d50.to_csv(paths.out_dir / "qc_missing_d50_2009_2030.csv", index=False)


if __name__ == "__main__":
    main()
