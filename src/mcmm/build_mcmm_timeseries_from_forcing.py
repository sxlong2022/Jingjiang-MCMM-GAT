import argparse
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

import wd_delayed_ssm


def _seed_from(scenario: str, member: int, user_seed: Optional[int]) -> int:
    if user_seed is not None:
        return int(user_seed)
    s = str(scenario)
    scen_part = sum((i + 1) * ord(ch) for i, ch in enumerate(s)) % 100000
    return int(scen_part + 1000003 * int(member))


def _fit_powerlaw(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if int(ok.sum()) < 5:
        return 1.0, 0.0
    lx = np.log(x[ok])
    ly = np.log(y[ok])
    b, a = np.polyfit(lx, ly, 1)
    a = float(np.exp(a))
    b = float(b)
    if not np.isfinite(a) or a <= 0:
        a = 1.0
    if not np.isfinite(b):
        b = 0.0
    return a, b


def _load_q_reference(master_csv: Path) -> dict[int, float]:
    df = pd.read_csv(master_csv)
    if not {"month", "Q"}.issubset(df.columns):
        raise ValueError(f"Invalid master timeseries columns: {master_csv}")
    m = df["month"].astype(str).str.replace(".0", "", regex=False)
    df = df.assign(month_str=m)
    df = df[df["month_str"].str.len() == 6].copy()
    df["m"] = df["month_str"].str[4:6].astype(int)
    q = pd.to_numeric(df["Q"], errors="coerce")
    q_clim = df.assign(Q_num=q).groupby("m")["Q_num"].mean().to_dict()
    q_clim = {int(k): float(v) for k, v in q_clim.items() if np.isfinite(v) and float(v) > 0}
    return q_clim


def _parse_yyyymm(yyyymm: str) -> pd.Timestamp:
    yyyymm = str(yyyymm).strip()
    if len(yyyymm) != 6 or not yyyymm.isdigit():
        raise ValueError(f"Invalid yyyymm: {yyyymm}")
    y = int(yyyymm[:4])
    m = int(yyyymm[4:6])
    return pd.Timestamp(year=y, month=m, day=1)


def _fmt_yyyymm(ts: pd.Timestamp) -> str:
    return f"{ts.year:04d}{ts.month:02d}"


def _month_range(start_yyyymm: str, end_yyyymm: str) -> list[str]:
    start = _parse_yyyymm(start_yyyymm)
    end = _parse_yyyymm(end_yyyymm)
    if end < start:
        raise ValueError(f"Invalid range: {start_yyyymm} ~ {end_yyyymm}")
    months = pd.date_range(start=start, end=end, freq="MS")
    return [_fmt_yyyymm(d) for d in months]


def _shift_yyyymm(yyyymm: str, delta_months: int) -> str:
    ts = _parse_yyyymm(yyyymm)
    ts2 = ts + pd.DateOffset(months=int(delta_months))
    return _fmt_yyyymm(ts2)


def _load_monthly_climatology(width_stats_csv: Path, depth_stats_csv: Path) -> tuple[dict[int, float], dict[int, float]]:
    dfw = pd.read_csv(width_stats_csv)
    if not {"Month", "Mean_Width_Meters"}.issubset(dfw.columns):
        raise ValueError(f"Invalid width stats csv columns: {width_stats_csv}")
    w_clim = dfw.groupby("Month")["Mean_Width_Meters"].mean().to_dict()

    dfd = pd.read_csv(depth_stats_csv)

    if {"Month", "Mean_Depth_Along"}.issubset(dfd.columns):
        d_clim = dfd.groupby("Month")["Mean_Depth_Along"].mean().to_dict()
    elif {"Month", "Mean_Depth"}.issubset(dfd.columns):
        d_clim = dfd.groupby("Month")["Mean_Depth"].mean().to_dict()
    elif {"month", "D"}.issubset(dfd.columns):
        m = dfd["month"].astype(str).str.replace(".0", "", regex=False)
        dfd = dfd.assign(month_str=m)
        dfd = dfd[dfd["month_str"].str.len() == 6].copy()
        dfd["Month"] = dfd["month_str"].str[4:6].astype(int)
        dfd["D_num"] = pd.to_numeric(dfd["D"], errors="coerce")
        d_clim = dfd.groupby("Month")["D_num"].mean().to_dict()
    else:
        raise ValueError(f"Invalid depth stats csv columns: {depth_stats_csv}")

    return ({int(k): float(v) for k, v in w_clim.items()}, {int(k): float(v) for k, v in d_clim.items()})


def _load_e_rate_reference(master_csv: Path) -> tuple[float, dict[int, float]]:
    df = pd.read_csv(master_csv)
    if not {"month", "E_rate"}.issubset(df.columns):
        raise ValueError(f"Invalid master timeseries columns: {master_csv}")

    m = df["month"].astype(str).str.replace(".0", "", regex=False)
    df = df.assign(month_str=m)
    df = df[df["month_str"].str.len() == 6].copy()
    df["m"] = df["month_str"].str[4:6].astype(int)

    e = pd.to_numeric(df["E_rate"], errors="coerce")
    e_med = float(np.nanmedian(e.values))
    e_clim = df.assign(E_rate_num=e).groupby("m")["E_rate_num"].median().to_dict()
    e_clim = {int(k): float(v) for k, v in e_clim.items() if np.isfinite(v)}

    if not np.isfinite(e_med) or e_med <= 0:
        raise ValueError(f"Invalid E_rate median from {master_csv}: {e_med}")

    return e_med, e_clim


def _load_monthly_wd_bounds_from_master(
    master_csv: Path,
    q_low: float,
    q_high: float,
) -> tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float]]:
    df = pd.read_csv(master_csv)
    if not {"month", "W", "D"}.issubset(df.columns):
        raise ValueError(f"Invalid master timeseries columns: {master_csv}")

    m = df["month"].astype(str).str.replace(".0", "", regex=False)
    df = df.assign(month_str=m)
    df = df[df["month_str"].str.len() == 6].copy()
    df["month_num"] = df["month_str"].str[4:6].astype(int)

    w = pd.to_numeric(df["W"], errors="coerce")
    d = pd.to_numeric(df["D"], errors="coerce")
    df = df.assign(W_num=w, D_num=d)

    ql = float(q_low)
    qh = float(q_high)
    if not (0.0 <= ql < qh <= 1.0):
        raise ValueError(f"Invalid quantile bounds: q_low={q_low}, q_high={q_high}")

    g = df.groupby("month_num")
    w_lo = g["W_num"].quantile(ql).to_dict()
    w_hi = g["W_num"].quantile(qh).to_dict()
    d_lo = g["D_num"].quantile(ql).to_dict()
    d_hi = g["D_num"].quantile(qh).to_dict()

    def _clean(x: dict) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for k, v in x.items():
            try:
                kk = int(k)
                vv = float(v)
            except Exception:
                continue
            if np.isfinite(vv) and vv > 0:
                out[kk] = vv
        return out

    return _clean(w_lo), _clean(w_hi), _clean(d_lo), _clean(d_hi)


def _build_timeseries(
    forcing_csv: Path,
    scenario: str,
    member: int,
    start_yyyymm: str,
    end_yyyymm: str,
    w_clim: dict[int, float],
    d_clim: dict[int, float],
    wd_method: str,
    q_clim: dict[int, float],
    w_q_power_b: Union[float, Dict[int, float]],
    d_q_power_b: Union[float, Dict[int, float]],
    e_med: float,
    e_clim: dict[int, float],
    e_rate_method: str,
    e_rate_base: str,
    e_rate_q_power: float,
    e_rate_q_clip_min: float,
    e_rate_q_clip_max: float,
    rng_seed: int,
    wd_noise_ln_sigma: float,
    e_rate_noise_ln_sigma: float,
    wd_clip_ref: str,
    wd_clip_q_low: float,
    wd_clip_q_high: float,
    wd_clip_pad_frac: float,
    wd_clip_w_lo: Dict[int, float],
    wd_clip_w_hi: Dict[int, float],
    wd_clip_d_lo: Dict[int, float],
    wd_clip_d_hi: Dict[int, float],
    physical_ds_default: float,
    wd_clip_diag_csv: Optional[Path],
    wd_ssm_params: Optional[dict],
    wd_ssm_driver_hist: Optional[pd.DataFrame],
    wd_ssm_logw_prev: float,
    wd_ssm_logd_prev: float,
    wd_ssm_sigma_scale: float,
    wd_ssm_sample: bool,
) -> pd.DataFrame:
    df = pd.read_csv(forcing_csv)

    need_cols = {"scenario", "member", "yyyymm", "Q_m3s"}
    missing = sorted([c for c in need_cols if c not in df.columns])
    if missing:
        raise ValueError(f"Missing columns in forcing csv {forcing_csv}: {missing}")

    df["scenario"] = df["scenario"].astype(str)
    df["member"] = pd.to_numeric(df["member"], errors="coerce").astype("Int64")
    df["yyyymm"] = df["yyyymm"].astype(str).str.replace(".0", "", regex=False)

    sub = df[(df["scenario"] == str(scenario)) & (df["member"] == int(member))].copy()
    if sub.empty:
        raise ValueError(f"No rows for scenario={scenario}, member={member} in {forcing_csv}")

    months = _month_range(start_yyyymm, end_yyyymm)
    sub = sub[sub["yyyymm"].isin(months)].copy()

    got = set(sub["yyyymm"].tolist())
    missing_months = [m for m in months if m not in got]
    if missing_months:
        head = ",".join(missing_months[:12])
        tail = "" if len(missing_months) <= 12 else f" ... (+{len(missing_months) - 12})"
        raise ValueError(f"forcing csv missing months for scenario={scenario}, member={member}: {head}{tail}")

    sub["month_num"] = sub["yyyymm"].str[4:6].astype(int)

    out = pd.DataFrame({"month": months})
    out["month_num"] = out["month"].str[4:6].astype(int)
    merge_cols = ["yyyymm", "Q_m3s", "D50_sample_m", "D50_m"]
    if "Qs" in sub.columns:
        merge_cols.append("Qs")
    out = out.merge(sub[merge_cols].rename(columns={"yyyymm": "month"}), on="month", how="left")

    out["Q"] = pd.to_numeric(out["Q_m3s"], errors="coerce")
    out["Qs"] = pd.to_numeric(out.get("Qs"), errors="coerce")

    rng = np.random.default_rng(int(rng_seed))

    wd_method = str(wd_method)
    if wd_method == "ahg":
        wd_method = "q_scaled"
    elif wd_method == "ahg_monthly":
        wd_method = "q_scaled_monthly"

    if wd_method == "climatology":
        out["W"] = out["month_num"].map(w_clim)
        out["D"] = out["month_num"].map(d_clim)
    elif wd_method in {"q_scaled", "q_scaled_monthly"}:
        base_w = out["month_num"].map(w_clim)
        base_d = out["month_num"].map(d_clim)
        base_q = out["month_num"].map(q_clim)
        base_q = pd.to_numeric(base_q, errors="coerce")
        q = pd.to_numeric(out["Q"], errors="coerce")
        ratio = q / base_q
        ratio = ratio.where(np.isfinite(ratio) & (ratio > 0), 1.0)

        if isinstance(w_q_power_b, dict):
            w_b = out["month_num"].map(w_q_power_b)
            w_b = pd.to_numeric(w_b, errors="coerce").fillna(0.0)
        else:
            w_b = float(w_q_power_b)
        if isinstance(d_q_power_b, dict):
            d_b = out["month_num"].map(d_q_power_b)
            d_b = pd.to_numeric(d_b, errors="coerce").fillna(0.0)
        else:
            d_b = float(d_q_power_b)

        out["W"] = pd.to_numeric(base_w, errors="coerce") * (ratio ** w_b)
        out["D"] = pd.to_numeric(base_d, errors="coerce") * (ratio ** d_b)
        out["W"] = out["W"].where(np.isfinite(out["W"]) & (out["W"] > 0), base_w)
        out["D"] = out["D"].where(np.isfinite(out["D"]) & (out["D"] > 0), base_d)
    elif wd_method == "delayed_ssm":
        if wd_ssm_params is None:
            raise ValueError("delayed_ssm requires wd_ssm_params")
        if wd_ssm_driver_hist is None or wd_ssm_driver_hist.empty:
            raise ValueError("delayed_ssm requires wd_ssm_driver_hist")

        if "Qs" not in out.columns:
            raise ValueError("delayed_ssm requires Qs column in forcing csv")
        if not np.isfinite(pd.to_numeric(out["Qs"], errors="coerce")).all():
            raise ValueError("delayed_ssm requires finite Qs for all months")

        d50 = pd.to_numeric(out.get("D50_sample_m"), errors="coerce")
        d50_fallback = pd.to_numeric(out.get("D50_m"), errors="coerce")
        d50_eff = d50.where(np.isfinite(d50), d50_fallback)
        d50_eff = d50_eff.fillna(float(physical_ds_default))

        hist = wd_ssm_driver_hist.copy()
        need_hist_cols = {"yyyymm", "Q", "Qs", "D50"}
        if not need_hist_cols.issubset(set(hist.columns)):
            raise ValueError(f"wd_ssm_driver_hist missing columns: {sorted(list(need_hist_cols - set(hist.columns)))}")

        fut = pd.DataFrame(
            {
                "yyyymm": out["month"].astype(str),
                "Q": pd.to_numeric(out["Q"], errors="coerce"),
                "Qs": pd.to_numeric(out["Qs"], errors="coerce"),
                "D50": pd.to_numeric(d50_eff, errors="coerce"),
            }
        )

        feat_all = pd.concat([hist, fut], ignore_index=True)
        feat_all = wd_delayed_ssm.build_wd_features(feat_all)
        feat_all["const"] = 1.0

        p_w = wd_ssm_params.get("W")
        p_d = wd_ssm_params.get("D")
        if not isinstance(p_w, dict) or not isinstance(p_d, dict):
            raise ValueError("Invalid wd_ssm_params: missing W/D")

        feat_fut = feat_all.iloc[len(hist) :].copy().reset_index(drop=True)

        feat_fut = wd_delayed_ssm.apply_feature_clip(feat_fut, p_w.get("clip", {}))
        z_w = list(p_w.get("z_cols") or [])
        if not z_w:
            raise ValueError("Invalid wd_ssm_params.W.z_cols")
        miss_w = [c for c in z_w if c not in feat_fut.columns]
        if miss_w:
            raise ValueError(f"delayed_ssm missing W feature columns: {miss_w}")
        mu_w = feat_fut.loc[:, z_w].to_numpy(dtype=float) @ np.asarray(p_w.get("theta"), dtype=float)
        y_w = wd_delayed_ssm.simulate_delayed_ssm_from_prev(
            mu=mu_w,
            phi=float(p_w.get("phi")),
            beta=float(p_w.get("beta")),
            y_prev=float(wd_ssm_logw_prev),
            sigma=float(p_w.get("sigma")) * float(wd_ssm_sigma_scale),
            rng=rng,
            sample=bool(wd_ssm_sample),
        )

        feat_fut = wd_delayed_ssm.apply_feature_clip(feat_fut, p_d.get("clip", {}))
        z_d = list(p_d.get("z_cols") or [])
        if not z_d:
            raise ValueError("Invalid wd_ssm_params.D.z_cols")
        miss_d = [c for c in z_d if c not in feat_fut.columns]
        if miss_d:
            raise ValueError(f"delayed_ssm missing D feature columns: {miss_d}")
        mu_d = feat_fut.loc[:, z_d].to_numpy(dtype=float) @ np.asarray(p_d.get("theta"), dtype=float)
        y_d = wd_delayed_ssm.simulate_delayed_ssm_from_prev(
            mu=mu_d,
            phi=float(p_d.get("phi")),
            beta=float(p_d.get("beta")),
            y_prev=float(wd_ssm_logd_prev),
            sigma=float(p_d.get("sigma")) * float(wd_ssm_sigma_scale),
            rng=rng,
            sample=bool(wd_ssm_sample),
        )

        out["W"] = np.exp(y_w)
        out["D"] = np.exp(y_d)

        base_w = out["month_num"].map(w_clim)
        base_d = out["month_num"].map(d_clim)
        out["W"] = pd.to_numeric(out["W"], errors="coerce").where(np.isfinite(out["W"]) & (out["W"] > 0), base_w)
        out["D"] = pd.to_numeric(out["D"], errors="coerce").where(np.isfinite(out["D"]) & (out["D"] > 0), base_d)
    else:
        raise ValueError(f"Unknown wd_method: {wd_method}")

    wd_noise_ln_sigma = float(wd_noise_ln_sigma)
    if np.isfinite(wd_noise_ln_sigma) and wd_noise_ln_sigma > 0:
        eps_w = rng.normal(loc=0.0, scale=wd_noise_ln_sigma, size=len(out))
        eps_d = rng.normal(loc=0.0, scale=wd_noise_ln_sigma, size=len(out))
        out["W"] = pd.to_numeric(out["W"], errors="coerce") * np.exp(eps_w)
        out["D"] = pd.to_numeric(out["D"], errors="coerce") * np.exp(eps_d)

    w_before_clip = pd.to_numeric(out["W"], errors="coerce").copy()
    d_before_clip = pd.to_numeric(out["D"], errors="coerce").copy()
    w_lo = pd.Series(np.nan, index=out.index)
    w_hi = pd.Series(np.nan, index=out.index)
    d_lo = pd.Series(np.nan, index=out.index)
    d_hi = pd.Series(np.nan, index=out.index)

    wd_clip_ref = str(wd_clip_ref)
    if wd_clip_ref != "none":
        pad = float(wd_clip_pad_frac)
        if not (np.isfinite(pad) and pad >= 0):
            pad = 0.0

        w_lo = out["month_num"].map(wd_clip_w_lo)
        w_hi = out["month_num"].map(wd_clip_w_hi)
        d_lo = out["month_num"].map(wd_clip_d_lo)
        d_hi = out["month_num"].map(wd_clip_d_hi)

        w_lo = pd.to_numeric(w_lo, errors="coerce")
        w_hi = pd.to_numeric(w_hi, errors="coerce")
        d_lo = pd.to_numeric(d_lo, errors="coerce")
        d_hi = pd.to_numeric(d_hi, errors="coerce")

        w_lo = w_lo * (1.0 - pad)
        w_hi = w_hi * (1.0 + pad)
        d_lo = d_lo * (1.0 - pad)
        d_hi = d_hi * (1.0 + pad)

        w_lo = w_lo.where(np.isfinite(w_lo) & (w_lo > 0), -np.inf)
        w_hi = w_hi.where(np.isfinite(w_hi) & (w_hi > 0), np.inf)
        d_lo = d_lo.where(np.isfinite(d_lo) & (d_lo > 0), -np.inf)
        d_hi = d_hi.where(np.isfinite(d_hi) & (d_hi > 0), np.inf)

        out["W"] = pd.to_numeric(out["W"], errors="coerce").clip(lower=w_lo, upper=w_hi)
        out["D"] = pd.to_numeric(out["D"], errors="coerce").clip(lower=d_lo, upper=d_hi)

    if wd_clip_diag_csv is not None:
        w_after_clip = pd.to_numeric(out["W"], errors="coerce").copy()
        d_after_clip = pd.to_numeric(out["D"], errors="coerce").copy()
        clipped_w = ~pd.Series(
            np.isclose(w_before_clip.to_numpy(), w_after_clip.to_numpy(), equal_nan=True),
            index=out.index,
        )
        clipped_d = ~pd.Series(
            np.isclose(d_before_clip.to_numpy(), d_after_clip.to_numpy(), equal_nan=True),
            index=out.index,
        )
        diag = pd.DataFrame(
            {
                "month": out["month"],
                "W_before": w_before_clip,
                "W_after": w_after_clip,
                "W_lo": w_lo,
                "W_hi": w_hi,
                "W_clipped": clipped_w.astype(int),
                "D_before": d_before_clip,
                "D_after": d_after_clip,
                "D_lo": d_lo,
                "D_hi": d_hi,
                "D_clipped": clipped_d.astype(int),
            }
        )
        wd_clip_diag_csv = Path(wd_clip_diag_csv)
        wd_clip_diag_csv.parent.mkdir(parents=True, exist_ok=True)
        diag.to_csv(wd_clip_diag_csv, index=False)

    if e_rate_method in {"median", "monthly_median"}:
        if e_rate_method == "median":
            out["E_rate"] = float(e_med)
        else:
            out["E_rate"] = out["month_num"].map(e_clim)
            out["E_rate"] = out["E_rate"].fillna(float(e_med))
    elif e_rate_method == "q_scaled":
        if not q_clim:
            raise ValueError("q_scaled E_rate requires q_clim loaded from master_timeseries")

        if e_rate_base == "median":
            base_e = pd.Series(float(e_med), index=out.index)
        elif e_rate_base == "monthly_median":
            base_e = out["month_num"].map(e_clim)
            base_e = base_e.fillna(float(e_med))
        else:
            raise ValueError(f"Unknown e_rate_base: {e_rate_base}")

        base_q = out["month_num"].map(q_clim)
        base_q = pd.to_numeric(base_q, errors="coerce")
        q = pd.to_numeric(out["Q"], errors="coerce")
        ratio = q / base_q
        ratio = ratio.where(np.isfinite(ratio) & (ratio > 0), 1.0)

        clip_lo = float(e_rate_q_clip_min)
        clip_hi = float(e_rate_q_clip_max)
        if np.isfinite(clip_lo) and np.isfinite(clip_hi) and clip_hi >= clip_lo and clip_hi > 0:
            ratio = ratio.clip(lower=clip_lo, upper=clip_hi)

        out["E_rate"] = pd.to_numeric(base_e, errors="coerce") * (ratio ** float(e_rate_q_power))
        out["E_rate"] = out["E_rate"].where(np.isfinite(out["E_rate"]) & (out["E_rate"] > 0), base_e)
    else:
        raise ValueError(f"Unknown e_rate_method: {e_rate_method}")

    e_rate_noise_ln_sigma = float(e_rate_noise_ln_sigma)
    if np.isfinite(e_rate_noise_ln_sigma) and e_rate_noise_ln_sigma > 0:
        eps_e = rng.normal(loc=0.0, scale=e_rate_noise_ln_sigma, size=len(out))
        out["E_rate"] = pd.to_numeric(out["E_rate"], errors="coerce") * np.exp(eps_e)

    d50 = pd.to_numeric(out.get("D50_sample_m"), errors="coerce")
    d50_fallback = pd.to_numeric(out.get("D50_m"), errors="coerce")
    ds = d50.where(np.isfinite(d50), d50_fallback)
    ds = ds.fillna(float(physical_ds_default))
    out["physical_ds"] = ds

    core = ["Q", "W", "D", "E_rate", "physical_ds"]
    for c in core:
        bad = ~np.isfinite(pd.to_numeric(out[c], errors="coerce"))
        if bool(bad.any()):
            raise ValueError(f"Non-finite values in column {c} for months: {out.loc[bad, 'month'].head(12).tolist()}")

    return out[["month", "Q", "W", "D", "E_rate", "physical_ds"]]


def main():
    parser = argparse.ArgumentParser(description="Build MCMM monthly timeseries CSV from forcing scenarios")
    parser.add_argument("--start", default="202501", help="Start month yyyymm (inclusive)")
    parser.add_argument("--end", default="203012", help="End month yyyymm (inclusive)")
    parser.add_argument("--scenario", default="S0")
    parser.add_argument("--member", type=int, default=0)

    parser.add_argument(
        "--forcing_csv",
        default=None,
        help="Path to forcing_q_d50_scenarios_*.csv (default: data/processed/forcing_q_d50_scenarios_<start>_<end>.csv)",
    )
    parser.add_argument(
        "--master_timeseries",
        default=None,
        help="Reference historical MCMM timeseries (default: input/jingjiang_monthly_2016_2024_final.csv)",
    )
    parser.add_argument(
        "--e_rate_method",
        default="median",
        choices=["median", "monthly_median", "q_scaled"],
    )
    parser.add_argument(
        "--e_rate_base",
        default="monthly_median",
        choices=["median", "monthly_median"],
    )
    parser.add_argument("--e_rate_q_power", type=float, default=1.0)
    parser.add_argument("--e_rate_q_clip_min", type=float, default=0.5)
    parser.add_argument("--e_rate_q_clip_max", type=float, default=2.0)
    parser.add_argument(
        "--wd_method",
        default="climatology",
        choices=["climatology", "q_scaled", "ahg", "ahg_monthly", "delayed_ssm"],
    )

    parser.add_argument(
        "--wd_ssm_params",
        default=None,
        help="Path to delayed SSM W/D model params json (only used when --wd_method delayed_ssm).",
    )
    parser.add_argument(
        "--wd_ssm_sigma_scale",
        type=float,
        default=1.0,
        help="Scale factor applied to delayed SSM process noise sigma (default: 1.0).",
    )
    parser.add_argument(
        "--wd_ssm_sample",
        type=int,
        default=1,
        help="Whether to sample process noise in delayed SSM (0/1, default: 1).",
    )

    parser.add_argument(
        "--rng_seed",
        type=int,
        default=None,
        help="Optional RNG seed for member noise (default: derived from scenario+member).",
    )
    parser.add_argument(
        "--wd_noise_ln_sigma",
        type=float,
        default=0.0,
        help="Lognormal noise sigma applied to W and D after generation (default: 0).",
    )
    parser.add_argument(
        "--e_rate_noise_ln_sigma",
        type=float,
        default=0.0,
        help="Lognormal noise sigma applied to E_rate after generation (default: 0).",
    )

    parser.add_argument(
        "--wd_clip_ref",
        default="none",
        choices=["none", "hist_monthly_p05_p95"],
        help="Optional stability clip for generated W/D. Uses monthly bounds from master_timeseries (default: none).",
    )
    parser.add_argument("--wd_clip_q_low", type=float, default=0.05)
    parser.add_argument("--wd_clip_q_high", type=float, default=0.95)
    parser.add_argument(
        "--wd_clip_pad_frac",
        type=float,
        default=0.05,
        help="Padding fraction applied to W/D clip bounds (default: 0.05).",
    )
    parser.add_argument(
        "--wd_clip_diag_csv",
        default=None,
        help="Optional path to output W/D clip diagnostics csv (default: none).",
    )
    parser.add_argument("--physical_ds_default", type=float, default=0.00025)

    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: input/future_timeseries)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output csv path (default: <out_dir>/jingjiang_monthly_<start>_<end>_<scenario>_m<member>.csv)",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_raw_dir = repo_root / "data" / "raw"
    data_processed_dir = repo_root / "data" / "processed"
    width_stats_csv = data_raw_dir / "monthly_width_stats.csv"

    forcing_csv = (
        Path(args.forcing_csv)
        if args.forcing_csv
        else (data_processed_dir / f"forcing_q_d50_scenarios_{args.start}_{args.end}.csv")
    )

    master_csv = Path(args.master_timeseries) if args.master_timeseries else (data_raw_dir / "jingjiang_monthly_2016_2024_final.csv")

    out_dir = Path(args.out_dir) if args.out_dir else (data_processed_dir / "future_timeseries")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out) if args.out else (out_dir / f"jingjiang_monthly_{args.start}_{args.end}_{args.scenario}_m{int(args.member)}.csv")

    w_clim, d_clim = _load_monthly_climatology(width_stats_csv=width_stats_csv, depth_stats_csv=master_csv)
    e_med, e_clim = _load_e_rate_reference(master_csv=master_csv)

    wd_ssm_params = None
    wd_ssm_driver_hist: Optional[pd.DataFrame] = None
    wd_ssm_logw_prev = float("nan")
    wd_ssm_logd_prev = float("nan")

    if str(args.wd_method) == "delayed_ssm":
        p = Path(args.wd_ssm_params) if args.wd_ssm_params else (repo_root / "src" / "forcing" / "wd_ssm_params.json")
        wd_ssm_params = wd_delayed_ssm.load_params(p)

        prev2 = _shift_yyyymm(str(args.start), -2)
        prev1 = _shift_yyyymm(str(args.start), -1)

        df_master = pd.read_csv(master_csv)
        m = pd.Series(df_master.get("month")).astype(str).str.replace(".0", "", regex=False)
        df_master = df_master.assign(month_str=m)
        df_master = df_master[df_master["month_str"].str.len() == 6].copy()

        df_master = df_master.sort_values("month_str").reset_index(drop=True)
        by_m = df_master.set_index("month_str")

        ref_month = prev1 if prev1 in by_m.index else str(by_m.index[-1])
        w0 = float(pd.to_numeric(by_m.loc[ref_month, "W"], errors="coerce"))
        d0 = float(pd.to_numeric(by_m.loc[ref_month, "D"], errors="coerce"))
        if not (np.isfinite(w0) and w0 > 0 and np.isfinite(d0) and d0 > 0):
            raise ValueError(f"Invalid W/D initial state from master_timeseries at {ref_month}: W={w0}, D={d0}")
        wd_ssm_logw_prev = float(np.log(w0))
        wd_ssm_logd_prev = float(np.log(d0))

        qs_csv = data_raw_dir / "shashi_monthly_q_qs.csv"
        qs_df = wd_delayed_ssm.load_qs_monthly_csv(qs_csv)
        qs_df = qs_df.rename(columns={"Q_m3s": "Q_h", "Qs_kgm3": "Qs"})
        qs_df = qs_df.set_index("yyyymm")

        rows = []
        for mm in [prev2, prev1]:
            if mm not in by_m.index:
                raise ValueError(f"master_timeseries missing required history month for delayed_ssm: {mm}")
            q = float(pd.to_numeric(by_m.loc[mm, "Q"], errors="coerce"))
            d50 = float(pd.to_numeric(by_m.loc[mm, "physical_ds"], errors="coerce"))
            if mm not in qs_df.index:
                raise ValueError(f"Qs history csv missing month {mm}")
            qs = float(pd.to_numeric(qs_df.loc[mm, "Qs"], errors="coerce"))
            rows.append({"yyyymm": str(mm), "Q": q, "Qs": qs, "D50": d50})
        wd_ssm_driver_hist = pd.DataFrame(rows)

    q_clim = {}
    w_q_power_b = 0.0
    d_q_power_b = 0.0

    w_q_power_b_by_month: Dict[int, float] = {}
    d_q_power_b_by_month: Dict[int, float] = {}

    wd_clip_w_lo: Dict[int, float] = {}
    wd_clip_w_hi: Dict[int, float] = {}
    wd_clip_d_lo: Dict[int, float] = {}
    wd_clip_d_hi: Dict[int, float] = {}

    if str(args.wd_method) in {"q_scaled", "ahg", "ahg_monthly"} or str(args.e_rate_method) == "q_scaled":
        q_clim = _load_q_reference(master_csv=master_csv)

    if str(args.wd_method) in {"q_scaled", "ahg", "ahg_monthly"}:
        df_master = pd.read_csv(master_csv)
        m = df_master.get("month")
        m = pd.Series(m).astype(str).str.replace(".0", "", regex=False)
        df_master = df_master.assign(month_str=m)
        df_master = df_master[df_master["month_str"].str.len() == 6].copy()
        df_master["month_num"] = df_master["month_str"].str[4:6].astype(int)

        w_a, w_b = _fit_powerlaw(pd.to_numeric(df_master.get("Q"), errors="coerce"), pd.to_numeric(df_master.get("W"), errors="coerce"))
        d_a, d_b = _fit_powerlaw(pd.to_numeric(df_master.get("Q"), errors="coerce"), pd.to_numeric(df_master.get("D"), errors="coerce"))
        w_q_power_b = max(0.0, float(w_b))
        d_q_power_b = max(0.0, float(d_b))

        if str(args.wd_method) == "ahg_monthly":
            for mo in range(1, 13):
                subm = df_master[df_master["month_num"] == int(mo)]
                _wa, _wb = _fit_powerlaw(pd.to_numeric(subm.get("Q"), errors="coerce"), pd.to_numeric(subm.get("W"), errors="coerce"))
                _da, _db = _fit_powerlaw(pd.to_numeric(subm.get("Q"), errors="coerce"), pd.to_numeric(subm.get("D"), errors="coerce"))
                w_q_power_b_by_month[int(mo)] = max(0.0, float(_wb))
                d_q_power_b_by_month[int(mo)] = max(0.0, float(_db))
            for mo in range(1, 13):
                if not np.isfinite(w_q_power_b_by_month.get(int(mo), np.nan)):
                    w_q_power_b_by_month[int(mo)] = float(w_q_power_b)
                if not np.isfinite(d_q_power_b_by_month.get(int(mo), np.nan)):
                    d_q_power_b_by_month[int(mo)] = float(d_q_power_b)

    if str(args.wd_clip_ref) != "none":
        wd_clip_w_lo, wd_clip_w_hi, wd_clip_d_lo, wd_clip_d_hi = _load_monthly_wd_bounds_from_master(
            master_csv=master_csv,
            q_low=float(args.wd_clip_q_low),
            q_high=float(args.wd_clip_q_high),
        )

    rng_seed = _seed_from(scenario=str(args.scenario), member=int(args.member), user_seed=args.rng_seed)

    ts = _build_timeseries(
        forcing_csv=forcing_csv,
        scenario=str(args.scenario),
        member=int(args.member),
        start_yyyymm=str(args.start),
        end_yyyymm=str(args.end),
        w_clim=w_clim,
        d_clim=d_clim,
        wd_method=str(args.wd_method),
        q_clim=q_clim,
        w_q_power_b=(w_q_power_b_by_month if str(args.wd_method) == "ahg_monthly" else float(w_q_power_b)),
        d_q_power_b=(d_q_power_b_by_month if str(args.wd_method) == "ahg_monthly" else float(d_q_power_b)),
        e_med=e_med,
        e_clim=e_clim,
        e_rate_method=str(args.e_rate_method),
        e_rate_base=str(args.e_rate_base),
        e_rate_q_power=float(args.e_rate_q_power),
        e_rate_q_clip_min=float(args.e_rate_q_clip_min),
        e_rate_q_clip_max=float(args.e_rate_q_clip_max),
        rng_seed=int(rng_seed),
        wd_noise_ln_sigma=float(args.wd_noise_ln_sigma),
        e_rate_noise_ln_sigma=float(args.e_rate_noise_ln_sigma),
        wd_clip_ref=str(args.wd_clip_ref),
        wd_clip_q_low=float(args.wd_clip_q_low),
        wd_clip_q_high=float(args.wd_clip_q_high),
        wd_clip_pad_frac=float(args.wd_clip_pad_frac),
        wd_clip_w_lo=wd_clip_w_lo,
        wd_clip_w_hi=wd_clip_w_hi,
        wd_clip_d_lo=wd_clip_d_lo,
        wd_clip_d_hi=wd_clip_d_hi,
        physical_ds_default=float(args.physical_ds_default),
        wd_clip_diag_csv=(Path(args.wd_clip_diag_csv) if args.wd_clip_diag_csv else None),
        wd_ssm_params=wd_ssm_params,
        wd_ssm_driver_hist=wd_ssm_driver_hist,
        wd_ssm_logw_prev=float(wd_ssm_logw_prev),
        wd_ssm_logd_prev=float(wd_ssm_logd_prev),
        wd_ssm_sigma_scale=float(args.wd_ssm_sigma_scale),
        wd_ssm_sample=bool(int(args.wd_ssm_sample)),
    )

    ts.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")
    print(f"Rows: {len(ts)}")


if __name__ == "__main__":
    main()
