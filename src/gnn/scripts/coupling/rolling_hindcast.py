import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # GNN/
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from models.gat_model import MigrationGATEnhanced, MigrationGATSimple  # noqa: E402


def _parse_yyyymm(s: str) -> tuple[int, int]:
    s = str(s).strip()
    if len(s) != 6 or not s.isdigit():
        raise ValueError(f"Invalid yyyymm: {s}")
    y = int(s[:4])
    m = int(s[4:6])
    if not (1 <= m <= 12):
        raise ValueError(f"Invalid month in yyyymm: {s}")
    return y, m


def _fmt_yyyymm(y: int, m: int) -> str:
    return f"{y:04d}{m:02d}"


def _add_months(yyyymm: str, months: int) -> str:
    y, m = _parse_yyyymm(yyyymm)
    idx0 = y * 12 + (m - 1)
    idx = idx0 + int(months)
    yy = idx // 12
    mm = (idx % 12) + 1
    return _fmt_yyyymm(yy, mm)


def _align_centerline_orientation(ref_cl: np.ndarray, cand_cl: np.ndarray) -> np.ndarray:
    if ref_cl is None or cand_cl is None:
        return cand_cl
    if ref_cl.shape != cand_cl.shape:
        return cand_cl
    d1 = np.nanmean(np.linalg.norm(ref_cl - cand_cl, axis=1))
    d2 = np.nanmean(np.linalg.norm(ref_cl - cand_cl[::-1], axis=1))
    return cand_cl if d1 <= d2 else cand_cl[::-1]


def _compute_migration(cl_current: np.ndarray, cl_next: np.ndarray, mode: str = "min") -> np.ndarray:
    cl_next_aligned = _align_centerline_orientation(cl_current, cl_next)
    migration_corresp = np.linalg.norm(cl_next_aligned - cl_current, axis=1)

    tree = cKDTree(cl_next)
    migration_min = tree.query(cl_current, k=1)[0]

    if mode == "min":
        return migration_min

    m_corresp = float(np.nanmean(migration_corresp))
    m_min = float(np.nanmean(migration_min))
    if np.isfinite(m_corresp) and np.isfinite(m_min) and m_min > 1e-6 and (m_corresp / m_min) > 5.0:
        return migration_min
    return migration_corresp


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    ac = a - a.mean()
    bc = b - b.mean()
    denom = (np.sqrt(np.sum(ac**2) * np.sum(bc**2)) + 1e-12)
    return float(np.sum(ac * bc) / denom)


def _pointwise_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.linalg.norm(a - b, axis=1)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "pw_mean": float(np.mean(d)),
        "pw_p95": float(np.percentile(d, 95)),
        "pw_max": float(np.max(d)),
    }


def _nn_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    tree = cKDTree(b)
    d = tree.query(a, k=1)[0]
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "nn_mean": float(np.mean(d)),
        "nn_p95": float(np.percentile(d, 95)),
        "nn_max": float(np.max(d)),
    }


def _project_disp_to_normal(centerline: np.ndarray, disp_vec: np.ndarray) -> np.ndarray:
    centerline = np.asarray(centerline, dtype=float)
    disp_vec = np.asarray(disp_vec, dtype=float)

    if centerline.ndim != 2 or centerline.shape[1] != 2:
        return disp_vec
    if disp_vec.shape != centerline.shape:
        return disp_vec

    n = int(centerline.shape[0])
    if n < 2:
        return disp_vec

    tang = np.zeros_like(centerline, dtype=float)
    tang[0] = centerline[1] - centerline[0]
    tang[-1] = centerline[-1] - centerline[-2]
    if n > 2:
        tang[1:-1] = centerline[2:] - centerline[:-2]

    tang_norm = np.linalg.norm(tang, axis=1)
    ok = tang_norm > 1e-12

    tang_unit = np.zeros_like(tang, dtype=float)
    tang_unit[ok] = tang[ok] / tang_norm[ok, None]

    normal = np.stack([-tang_unit[:, 1], tang_unit[:, 0]], axis=1)
    dot = np.sum(disp_vec * normal, axis=1, keepdims=True)
    disp_proj = dot * normal

    disp_out = np.array(disp_vec, copy=True)
    disp_out[ok] = disp_proj[ok]
    return disp_out


def _load_master_timeseries_map(master_csv: Path) -> dict[str, list]:
    """Load the master monthly timeseries once to avoid repeated disk I/O.

    Returns:
        dict: yyyymm(str) -> [month, Q, W, D, E_rate, physical_ds]
    """
    if not master_csv.exists():
        raise FileNotFoundError(f"Master timeseries not found: {master_csv}")

    out = {}
    with master_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = ["month", "Q", "W", "D", "E_rate", "physical_ds"]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing columns in {master_csv}: {missing}")

        for row in reader:
            m_raw = str(row.get("month", "")).strip()
            m_raw = m_raw.replace(".0", "")
            if not m_raw.isdigit() or len(m_raw) != 6:
                continue
            out[m_raw] = [
                m_raw,
                row.get("Q", ""),
                row.get("W", ""),
                row.get("D", ""),
                row.get("E_rate", ""),
                row.get("physical_ds", ""),
            ]
    return out


def _load_scenario_timeseries_map(scenario_csv: Path) -> dict[str, dict]:
    if not scenario_csv.exists():
        raise FileNotFoundError(f"Scenario timeseries not found: {scenario_csv}")

    out = {}
    with scenario_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = ["month", "Q", "W", "D", "E_rate", "physical_ds"]
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing columns in {scenario_csv}: {missing}")

        for row in reader:
            m_raw = str(row.get("month", "")).strip().replace(".0", "")
            if not m_raw.isdigit() or len(m_raw) != 6:
                continue
            out[m_raw] = dict(row)
    return out


def _try_float(v, default: float = float("nan")) -> float:
    try:
        if v is None:
            return float(default)
        s = str(v).strip()
        if s == "":
            return float(default)
        return float(s)
    except Exception:
        return float(default)


def _rowlist_to_dict(ts_row) -> dict:
    if ts_row is None:
        return {}
    try:
        m, q, w, d, e, ds = list(ts_row)[:6]
    except Exception:
        return {}
    return {
        "month": str(m),
        "Q": q,
        "W": w,
        "D": d,
        "E_rate": e,
        "physical_ds": ds,
    }


def _load_along_vector_csv(csv_fp: Optional[Path], month: str, n_expected: int = 1000) -> Optional[np.ndarray]:
    if csv_fp is None:
        return None
    csv_fp = Path(csv_fp)
    if not csv_fp.exists():
        return None

    try:
        with csv_fp.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return None
            if month not in header:
                return None
            j = header.index(month)
            vals = []
            for row in reader:
                if not row:
                    continue
                if j >= len(row):
                    vals.append(float("nan"))
                    continue
                vals.append(_try_float(row[j], default=float("nan")))
        arr = np.asarray(vals, dtype=float).reshape(-1)
        if arr.size != int(n_expected):
            return None
        return arr
    except Exception:
        return None


def _parse_simulation_dat(simulation_dat: Path):
    rows = []
    if not simulation_dat.exists():
        return rows
    for line in simulation_dat.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("jp") or line.startswith("-"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            jp = int(parts[0])
            jt = int(parts[1])
            t = float(parts[2])
            n = int(parts[3])
        except ValueError:
            continue
        rows.append({"jp": jp, "jt": jt, "time": t, "n": n})
    return rows


def _read_configuration_xy_nd(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"Invalid configuration file format: {path}")
    xy = data[:, :2]
    mask = np.isfinite(xy).all(axis=1)
    xy = xy[mask]
    if len(xy) < 2:
        raise ValueError(f"Too few valid points in {path}")
    xy = _dedup_coords(xy, eps_m=1e-12)
    if len(xy) < 2:
        raise ValueError(f"Too few non-duplicate points in {path}")
    return xy


def _resample_to_n_points(xy: np.ndarray, n_out: int) -> np.ndarray:
    if len(xy) < 2:
        raise ValueError("Too few points")
    d = np.sqrt(np.sum((xy[1:] - xy[:-1]) ** 2, axis=1))
    s = np.concatenate(([0.0], np.cumsum(d)))
    total = float(s[-1])
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid arclength")
    target_s = np.linspace(0.0, total, int(n_out))
    x = np.interp(target_s, s, xy[:, 0])
    y = np.interp(target_s, s, xy[:, 1])
    return np.column_stack([x, y])


def _parse_mcmm_centerline_utm(
    run_output_dir: Path,
    simname: str,
    start_yyyymm: str,
    target_yyyymm: str,
    w_ref: float,
    origin_x: float,
    origin_y: float,
    n_out: int = 1000,
) -> np.ndarray:
    """Parse a single target-month centerline from MCMM output without spawning a subprocess."""
    sim_dat = run_output_dir / f"{simname}_simulation.dat"
    rows = _parse_simulation_dat(sim_dat)

    jp_target = None
    for r in rows:
        t = r.get("time", np.nan)
        if not np.isfinite(t):
            continue
        months = int(round(float(t) * 12.0))
        yyyymm = _add_months(start_yyyymm, months)
        if yyyymm == target_yyyymm:
            jp_target = int(r["jp"])
            break

    if jp_target is None:
        config_files = []
        for p in run_output_dir.glob("configuration_*.out"):
            stem = p.stem
            try:
                jp = int(stem.split("_")[-1])
            except Exception:
                continue
            config_files.append((jp, p))
        if not config_files:
            raise FileNotFoundError(f"No configuration_*.out found in {run_output_dir}")
        config_files.sort(key=lambda t: t[0])
        jp_target, cfg_file = config_files[-1]
    else:
        cfg_file = run_output_dir / f"configuration_{jp_target:03d}.out"
        if not cfg_file.exists():
            cfg_file = run_output_dir / f"configuration_{jp_target}.out"
        if not cfg_file.exists():
            candidates = sorted(run_output_dir.glob(f"configuration_*{jp_target}*.out"))
            if candidates:
                cfg_file = candidates[0]
            else:
                raise FileNotFoundError(f"Target configuration file not found for jp={jp_target} in {run_output_dir}")

    xy_nd = _read_configuration_xy_nd(cfg_file)
    xy_nd_1000 = _resample_to_n_points(xy_nd, n_out)
    xy_utm = np.empty_like(xy_nd_1000)
    xy_utm[:, 0] = xy_nd_1000[:, 0] * w_ref + origin_x
    xy_utm[:, 1] = xy_nd_1000[:, 1] * w_ref + origin_y
    return xy_utm


def _dedup_coords(coords: np.ndarray, eps_m: float = 1e-6) -> np.ndarray:
    if len(coords) < 2:
        return coords
    d = np.sqrt(np.sum((coords[1:] - coords[:-1]) ** 2, axis=1))
    keep = np.concatenate(([True], d > eps_m))
    return coords[keep]


def _resample_by_arclength(xy: np.ndarray, target_deltas: float) -> np.ndarray:
    if len(xy) < 2:
        raise ValueError("Too few points")
    d = np.sqrt(np.sum((xy[1:] - xy[:-1]) ** 2, axis=1))
    s = np.concatenate(([0.0], np.cumsum(d)))
    total = float(s[-1])
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Invalid arclength: {total}")

    n_out = int(round(total / float(target_deltas))) + 1
    if n_out < 2:
        n_out = 2
    target_s = np.linspace(0.0, total, n_out)
    x = np.interp(target_s, s, xy[:, 0])
    y = np.interp(target_s, s, xy[:, 1])
    return np.column_stack([x, y])


def _utm_to_xy_nd(utm_xy: np.ndarray, w_ref: float, origin_x: float, origin_y: float) -> np.ndarray:
    xy = np.empty_like(utm_xy)
    xy[:, 0] = (utm_xy[:, 0] - origin_x) / w_ref
    xy[:, 1] = (utm_xy[:, 1] - origin_y) / w_ref
    return xy


def _load_transform(
    mcmm_input_dir: Path,
    initial_month: str,
    obs_centerline_dir: Optional[Path],
    init_centerline_fp: Optional[Path] = None,
    w_ref_default: float = 880.0,
):
    meta_file = mcmm_input_dir / "jingjiang_centerline_transform.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        return float(meta["w_ref"]), float(meta["origin_utm_x"]), float(meta["origin_utm_y"])

    fp = None
    if init_centerline_fp is not None:
        fp = Path(init_centerline_fp)
    elif obs_centerline_dir is not None:
        fp = Path(obs_centerline_dir) / f"{initial_month}_centerline_geo_1000pts.npy"

    if fp is None:
        raise FileNotFoundError("Transform json not found, and no centerline provided for deriving origin.")
    if not fp.exists():
        raise FileNotFoundError(f"Transform json not found, and initial centerline not found: {fp}")
    coords = np.asarray(np.load(fp), dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Invalid centerline array shape: {coords.shape}")
    return float(w_ref_default), float(coords[0, 0]), float(coords[0, 1])


def _centerline_utm_to_xy_nd(
    coords_utm: np.ndarray,
    w_ref: float,
    origin_x: float,
    origin_y: float,
    n0_fixed=None,
) -> np.ndarray:
    coords = np.asarray(coords_utm, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Invalid centerline array shape: {coords.shape}")
    coords = _dedup_coords(coords)

    xy_nd = _utm_to_xy_nd(coords, w_ref=w_ref, origin_x=origin_x, origin_y=origin_y)
    xy_nd = _dedup_coords(xy_nd, eps_m=1e-12)

    n0_int = None
    try:
        if n0_fixed is not None:
            n0_int = int(n0_fixed)
    except Exception:
        n0_int = None

    if n0_int is not None and n0_int > 1:
        xy_nd = _resample_to_n_points(xy_nd, n0_int)
    else:
        xy_nd = _resample_by_arclength(xy_nd, target_deltas=1.0)
    return xy_nd


def _write_centerline_xy(run_input_dir: Path, coords_utm: np.ndarray, w_ref: float, origin_x: float, origin_y: float, n0_fixed=None):
    xy_nd = _centerline_utm_to_xy_nd(
        coords_utm=coords_utm,
        w_ref=w_ref,
        origin_x=origin_x,
        origin_y=origin_y,
        n0_fixed=n0_fixed,
    )

    out_xy = run_input_dir / "jingjiang_centerline.xy"
    np.savetxt(out_xy, xy_nd, fmt="%.6f", delimiter="\t")
    return out_xy, int(xy_nd.shape[0])


def _replace_value_keep_comment(line: str, new_value: str) -> str:
    if "!" not in line:
        return str(new_value).rstrip() + "\n"
    pre, comment = line.split("!", 1)
    indent = ""
    for ch in pre:
        if ch in (" ", "\t"):
            indent += ch
        else:
            break
    return f"{indent}{str(new_value).rstrip()}    !{comment.rstrip()}\n"


def _make_one_step_sim(template_sim: Path, out_sim: Path, simname: str, n0: int, timeseries_file: str, dt0: str = "0.083333333333d0"):
    lines = template_sim.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)

    def _find_and_replace(key: str, value: str):
        for i, ln in enumerate(lines):
            if key in ln:
                lines[i] = _replace_value_keep_comment(ln, value)
                return True
        return False

    ok_simname = _find_and_replace("simulation name", simname)
    ok_n0 = _find_and_replace("! N0", str(int(n0)))
    ok_tts = _find_and_replace("! TTs", dt0)
    ok_dt0 = _find_and_replace("! dt0", dt0)
    ok_ifile = _find_and_replace("! ifile", "1")
    ok_ts = _find_and_replace("filetimeseries", timeseries_file)

    missing = []
    if not ok_simname:
        missing.append("simulation name")
    if not ok_n0:
        missing.append("N0")
    if not ok_tts:
        missing.append("TTs")
    if not ok_dt0:
        missing.append("dt0")
    if not ok_ifile:
        missing.append("ifile")
    if not ok_ts:
        missing.append("filetimeseries")

    if missing:
        raise RuntimeError(f"Template sim missing keys: {missing}")

    out_sim.write_text("".join(lines), encoding="utf-8")


def _extract_timeseries_row(master_csv: Path, yyyymm: str) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(master_csv)
    if "month" not in df.columns:
        raise ValueError(f"Missing 'month' column in {master_csv}")

    m_int = int(yyyymm)
    sub = df[df["month"].astype(int) == m_int].copy()
    if sub.empty:
        raise ValueError(f"Month {yyyymm} not found in {master_csv}")

    cols = ["month", "Q", "W", "D", "E_rate", "physical_ds"]
    missing = [c for c in cols if c not in sub.columns]
    if missing:
        raise ValueError(f"Missing columns in {master_csv}: {missing}")

    row = sub[cols].iloc[0]
    return row.to_numpy()


def _run_mcmm_once(mcmm_exe: Path, run_dir: Path, sim_filename: str):
    if not mcmm_exe.exists():
        raise FileNotFoundError(f"mcmm.exe not found: {mcmm_exe}")
    stdin_payload = f"{sim_filename}\n" + ("0\n" * 16)
    p = subprocess.run(
        [str(mcmm_exe)],
        input=stdin_payload,
        text=True,
        cwd=str(run_dir),
        capture_output=True,
    )
    if p.returncode != 0:
        sys.stderr.write(p.stdout)
        sys.stderr.write(p.stderr)
        raise RuntimeError(f"mcmm.exe failed (code={p.returncode}) at {run_dir}")


def _run_parse_output(parse_script: Path, run_output_dir: Path, simname: str, start_yyyymm: str, initial_month: str):
    cmd = [
        sys.executable,
        str(parse_script),
        "--start_yyyymm",
        start_yyyymm,
        "--initial_month",
        initial_month,
        "--simname",
        simname,
        "--output_dir",
        str(run_output_dir),
    ]
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout)
        sys.stderr.write(p.stderr)
        raise RuntimeError(f"parse_mcmm_output.py failed (code={p.returncode})")


def _pick_obs_centerline_dir(codes_dir: Path) -> Path:
    candidates = [
        codes_dir / "data" / "raw" / "monthly_centerlines"
    ]
    for d in candidates:
        if d.exists() and any(d.glob("*_centerline_geo_1000pts.npy")):
            return d
    return candidates[0]


def _load_fold_data(data_dir: Path, fold: int, tag: str, target: str, migration_mode: str, use_width: bool, use_depth: bool):
    suffix = f"_{tag}_{target}_{migration_mode}"
    if use_width:
        suffix += "_width"
    if use_depth:
        suffix += "_depth"
    train_file = data_dir / f"fold{fold}_train{suffix}.pt"
    val_file = data_dir / f"fold{fold}_val{suffix}.pt"
    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError(f"Missing dataset files: {train_file} or {val_file}")
    train_data = torch.load(train_file, weights_only=False)
    val_data = torch.load(val_file, weights_only=False)
    return train_data, val_data


def _load_model_from_cv_dir(model_dir: Path, fold: int, allow_negative_override: Optional[bool] = None):
    model_file = model_dir / f"fold{fold}_model.pt"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    ckpt = torch.load(model_file, weights_only=False, map_location="cpu")
    cfg = ckpt.get("config", {})
    model_type = cfg.get("model_type", "enhanced")

    use_width = bool(cfg.get("use_width", False))
    use_depth = bool(cfg.get("use_depth", False))
    use_mcmm = bool(cfg.get("use_mcmm", False))
    use_depth_mask = bool(cfg.get("use_depth_mask", False))

    hidden_channels = int(cfg.get("hidden_channels", 64))
    heads = int(cfg.get("heads", 4))
    dropout = float(cfg.get("dropout", 0.3))

    if "num_static" in cfg:
        num_static = int(cfg["num_static"])
    elif "model_state_dict" in ckpt and "film_gamma.weight" in ckpt["model_state_dict"]:
        num_static = ckpt["model_state_dict"]["film_gamma.weight"].shape[0]
    else:
        num_static = 14

    if "num_hydro" in cfg:
        num_hydro = int(cfg["num_hydro"])
    elif "model_state_dict" in ckpt and "film_gamma.weight" in ckpt["model_state_dict"]:
        num_hydro = ckpt["model_state_dict"]["film_gamma.weight"].shape[1]
    else:
        num_hydro = 7

    extra_node_dim = int(cfg.get("extra_node_dim", (1 if use_mcmm else 0) + (1 if use_depth_mask else 0)))
    in_channels = int(cfg.get("in_channels", (num_static + num_hydro + extra_node_dim)))

    allow_negative = bool(cfg.get("allow_negative", False))
    if allow_negative_override is not None:
        allow_negative = bool(allow_negative_override)

    if model_type == "simple":
        model = MigrationGATSimple(
            in_channels,
            hidden_channels=int(cfg.get("hidden_channels", 32)),
            heads=heads,
            dropout=dropout,
            use_width=use_width,
            use_depth=use_depth,
            allow_negative=allow_negative,
        )
    else:
        model = MigrationGATEnhanced(
            in_channels,
            hidden_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            num_static=num_static,
            num_hydro=num_hydro,
            use_width=use_width,
            use_depth=use_depth,
            allow_negative=allow_negative,
            extra_node_dim=extra_node_dim,
        )

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, cfg


def main():
    parser = argparse.ArgumentParser(description="Rolling hindcast (Scheme 1): 202301 start, 12-month rolling")

    parser.add_argument("--start", type=str, default="202301")
    parser.add_argument("--steps", type=int, default=12)

    parser.add_argument("--fold", type=int, default=5, help="Which fold dataset/model to use (default: 5 for hindcast 2023)")

    parser.add_argument("--data_dir", type=str, default=str(PROJECT_ROOT / "data" / "graph" / "cv_monthly_tf_ratio_min_ext"))
    parser.add_argument("--tag", type=str, default="tf")
    parser.add_argument("--target", type=str, default="ratio", choices=["obs", "residual", "ratio", "dn"])
    parser.add_argument("--migration_mode", type=str, default="min", choices=["corresp", "min"])
    parser.add_argument("--use_width", action="store_true")
    parser.add_argument("--use_depth", action="store_true")

    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "cv_monthly_results" / "ratio_min_enh_mcmm_wd_lp01_seed0_folds456"),
        help="Directory containing fold{n}_model.pt (e.g., outputs/cv_monthly_results/<run_tag>)",
    )

    parser.add_argument("--mcmm_root", type=str, default=None)
    parser.add_argument("--mcmm_exe", type=str, default=None)
    parser.add_argument("--template_sim", type=str, default=None)
    parser.add_argument("--master_timeseries", type=str, default=None)
    parser.add_argument("--obs_centerline_dir", type=str, default=None)

    parser.add_argument("--parse_mode", type=str, default="internal", choices=["internal", "script"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--profile", action="store_true")

    parser.add_argument(
        "--oracle_use_gt",
        action="store_true",
        help="Use dataset ground-truth target y as prediction (oracle), bypassing model inference. Useful for upper-bound diagnostics.",
    )

    parser.add_argument(
        "--oracle_dn_online",
        action="store_true",
        help="Only for --oracle_use_gt with --target=dn. Compute dn oracle from current-step geometry (mc1 vs obs1) instead of using dataset y.",
    )

    parser.add_argument(
        "--dagger_dump_dir",
        type=str,
        default=None,
        help="Optional: dump per-step training samples for DAgger (route1). Writes .pt files with x/edge_index/(x_width,x_depth)/y_expert.",
    )
    parser.add_argument(
        "--dagger_allow_nonempty",
        action="store_true",
        help="Allow writing DAgger samples into a non-empty --dagger_dump_dir (not recommended).",
    )

    parser.add_argument("--mode", type=str, default="rolling", choices=["rolling", "teacher", "forecast"])
    parser.add_argument("--scenario_timeseries", type=str, default=None)
    parser.add_argument("--init_centerline_fp", type=str, default=None)
    parser.add_argument("--width_along_csv", type=str, default=None)
    parser.add_argument("--depth_along_csv", type=str, default=None)
    parser.add_argument(
        "--rolling_update",
        type=str,
        default="pred",
        choices=["pred", "mcmm", "blend"],
        help="In rolling mode, update state using: pred=next_pred (default), mcmm=mc1 (no feedback of correction).",
    )
    parser.add_argument(
        "--rolling_blend_alpha",
        type=float,
        default=0.5,
        help="Only used when --rolling_update=blend. State update: current = (1-a)*mc1 + a*next_pred. a=0 -> mcmm, a=1 -> pred.",
    )

    parser.add_argument("--enable_redline", action="store_true")
    parser.add_argument("--redline_consecutive", type=int, default=2)
    parser.add_argument("--redline_alpha_reduced", type=float, default=0.2)
    parser.add_argument("--redline_l0_ratio_sat", type=float, default=0.20)
    parser.add_argument("--redline_l0_clip_frac", type=float, default=0.03)
    parser.add_argument("--redline_l1_ratio_sat", type=float, default=0.25)
    parser.add_argument("--redline_l1_clip_frac", type=float, default=0.05)
    parser.add_argument("--redline_l2_ratio_sat", type=float, default=0.35)
    parser.add_argument("--redline_l2_clip_frac", type=float, default=0.10)
    parser.add_argument("--redline_l2_roll_clip_instant", type=float, default=0.20)

    parser.add_argument(
        "--disp_project",
        type=str,
        default="none",
        choices=["none", "normal"],
        help="Project predicted displacement onto local normal direction before applying. Useful for stabilizing rolling feedback.",
    )

    parser.add_argument(
        "--mcmm_n0",
        type=int,
        default=0,
        help="Fix MCMM input centerline point count N0. Set <=0 to disable (default).",
    )
    parser.add_argument(
        "--lock_mcmm_n0",
        action="store_true",
        help="Lock MCMM input N0 to the initial month (computed from the starting centerline). Ignored if --mcmm_n0>0.",
    )

    parser.add_argument("--ratio_max", type=float, default=2.0)
    parser.add_argument(
        "--ratio_gain",
        type=float,
        default=1.0,
        help="Scale correction strength: ratio_eff = 1 + ratio_gain*(ratio_raw-1). ratio_gain=0 -> pure MCMM (ratio=1).",
    )
    parser.add_argument(
        "--dn_gain",
        type=float,
        default=1.0,
        help="When --target=dn, scale normal correction strength: dn_eff = dn_gain*dn_raw. dn_gain=0 -> pure MCMM.",
    )
    parser.add_argument(
        "--dn_clip",
        type=float,
        default=200.0,
        help="When --target=dn, clip predicted dn (meters): dn_eff = clip(dn_eff, -dn_clip, +dn_clip). Set <=0 to disable.",
    )
    parser.add_argument("--max_node_disp_m", type=float, default=500.0)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--keep_sandbox", action="store_true")

    parser.add_argument(
        "--sandbox_root",
        type=str,
        default=None,
        help="Optional sandbox root directory for MCMM runs. Useful on Windows to avoid long-path issues.",
    )

    args = parser.parse_args()

    start = str(args.start)
    steps = int(args.steps)
    if steps <= 0:
        raise ValueError("--steps must be positive")

    fold = int(args.fold)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    model_dir = Path(args.model_dir)
    if not bool(getattr(args, "oracle_use_gt", False)):
        if not model_dir.exists():
            raise FileNotFoundError(
                f"model_dir not found: {model_dir}. "
                "Please point --model_dir to a training output directory containing fold{n}_model.pt."
            )

    use_width = bool(args.use_width)
    use_depth = bool(args.use_depth)

    train_data, val_data = _load_fold_data(
        data_dir=data_dir,
        fold=fold,
        tag=args.tag,
        target=args.target,
        migration_mode=args.migration_mode,
        use_width=use_width,
        use_depth=use_depth,
    )

    month2src = {}
    if str(args.mode) != "forecast":
        for data_obj in (train_data, val_data):
            for i, mm in enumerate(list(getattr(data_obj, "months", []))):
                month2src[str(mm)] = (data_obj, int(i))

    codes_dir = PROJECT_ROOT.parent.parent
    obs_dir = Path(args.obs_centerline_dir) if args.obs_centerline_dir else _pick_obs_centerline_dir(codes_dir)
    if str(args.mode) != "forecast":
        if not obs_dir.exists():
            raise FileNotFoundError(f"Observed centerline dir not found: {obs_dir}")

    if str(args.mode) == "forecast":
        if not getattr(args, "init_centerline_fp", None):
            raise ValueError("--init_centerline_fp is required for --mode forecast")

    mcmm_root = Path(args.mcmm_root) if args.mcmm_root else (REPO_ROOT / "src" / "mcmm")
    mcmm_exe = Path(args.mcmm_exe) if args.mcmm_exe else (mcmm_root / "mcmm.exe")
    mcmm_input = mcmm_root
    parse_script = mcmm_root / "parse_mcmm_output.py"

    template_sim = Path(args.template_sim) if args.template_sim else (mcmm_root / "my_jingjiang.sim")
    master_ts = Path(args.master_timeseries) if args.master_timeseries else (REPO_ROOT / "data" / "raw" / "jingjiang_monthly_2016_2024_final.csv")

    if not template_sim.exists():
        raise FileNotFoundError(f"Template sim not found: {template_sim}")
    if not master_ts.exists():
        raise FileNotFoundError(f"Master timeseries not found: {master_ts}")
    if not parse_script.exists():
        raise FileNotFoundError(f"parse_mcmm_output.py not found: {parse_script}")

    ts_map = _load_master_timeseries_map(master_csv=master_ts)
    scenario_csv = Path(args.scenario_timeseries) if getattr(args, "scenario_timeseries", None) else None
    scenario_map = None
    if str(args.mode) == "forecast":
        if scenario_csv is None:
            raise ValueError("--scenario_timeseries is required for --mode forecast")
        scenario_map = _load_scenario_timeseries_map(scenario_csv)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    init_fp = None
    if str(args.mode) == "forecast":
        init_fp = Path(args.init_centerline_fp)
    elif getattr(args, "init_centerline_fp", None):
        init_fp = Path(args.init_centerline_fp)
    else:
        init_fp = obs_dir / f"{start}_centerline_geo_1000pts.npy"

    if not init_fp.exists():
        raise FileNotFoundError(f"Initial centerline not found: {init_fp}")

    w_ref, origin_x, origin_y = _load_transform(
        mcmm_input_dir=mcmm_input,
        initial_month=start,
        obs_centerline_dir=(obs_dir if str(args.mode) != "forecast" else None),
        init_centerline_fp=init_fp,
    )

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = (
            PROJECT_ROOT
            / "outputs"
            / "rolling_hindcast"
            / f"mode_{args.mode}"
            / f"start_{start}_steps{steps}_fold{fold}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = out_dir / "pred_centerlines"
    pred_dir.mkdir(parents=True, exist_ok=True)

    dagger_dump_dir = Path(args.dagger_dump_dir) if getattr(args, "dagger_dump_dir", None) else None
    if dagger_dump_dir is not None:
        dagger_dump_dir.mkdir(parents=True, exist_ok=True)
        if not bool(getattr(args, "dagger_allow_nonempty", False)):
            has_any = any(dagger_dump_dir.glob("*.pt")) or any(dagger_dump_dir.glob("*.json"))
            if has_any:
                raise FileExistsError(
                    f"dagger_dump_dir is not empty: {dagger_dump_dir}. "
                    "Refuse to overwrite. Use a new directory or pass --dagger_allow_nonempty."
                )

    sandbox_root = Path(args.sandbox_root) if args.sandbox_root else (out_dir / "sandbox")
    sandbox_root = sandbox_root.resolve()

    if os.name == "nt":
        test_name = f"hc_{start}_{_add_months(start, 1)}"
        test_fp = sandbox_root / test_name / "output" / f"{test_name}_simulation.dat"
        if len(str(test_fp)) > 240:
            sandbox_root = (Path(tempfile.gettempdir()) / f"mcmm_sandbox_s{start}_k{steps}_f{fold}").resolve()

    if sandbox_root.exists():
        if args.keep_sandbox:
            sandbox_root = (sandbox_root / f"keep_{start}_{int(time.time())}").resolve()
        else:
            shutil.rmtree(sandbox_root)
    sandbox_root.mkdir(parents=True, exist_ok=False)

    model = None
    model_cfg = {}
    edge_index = None
    x_static = None
    if not bool(getattr(args, "oracle_use_gt", False)):
        allow_negative_override = True if str(args.target) == "dn" else None
        model, model_cfg = _load_model_from_cv_dir(model_dir=model_dir, fold=fold, allow_negative_override=allow_negative_override)

        if bool(model_cfg.get("use_width", False)) and not bool(args.use_width):
            raise ValueError("Checkpoint requires --use_width, but it was not provided.")
        if bool(model_cfg.get("use_depth", False)) and not bool(args.use_depth):
            raise ValueError("Checkpoint requires --use_depth, but it was not provided.")

        model = model.to(device)
        edge_index = train_data.edge_index.to(device)
        x_static = train_data.x_static.to(device)

    current_cl = np.asarray(np.load(init_fp), dtype=float)
    ref_orient = np.asarray(current_cl, dtype=float)
    np.save(pred_dir / f"{start}_centerline_geo_1000pts.npy", current_cl)

    mcmm_n0_fixed = None
    if int(getattr(args, "mcmm_n0", 0)) > 1:
        mcmm_n0_fixed = int(getattr(args, "mcmm_n0"))
    elif bool(getattr(args, "lock_mcmm_n0", False)):
        try:
            xy0 = _centerline_utm_to_xy_nd(
                coords_utm=current_cl,
                w_ref=w_ref,
                origin_x=origin_x,
                origin_y=origin_y,
                n0_fixed=None,
            )
            mcmm_n0_fixed = int(xy0.shape[0])
        except Exception:
            mcmm_n0_fixed = None

    records = []

    redline_l1_streak = 0
    redline_l2_streak = 0

    q_hist = []
    last_q = float("nan")

    t_global0 = time.perf_counter()

    for k in range(steps):
        t_step0 = time.perf_counter()
        m0 = _add_months(start, k)
        m1 = _add_months(start, k + 1)

        if str(args.mode) == "teacher":
            obs0_fp = obs_dir / f"{m0}_centerline_geo_1000pts.npy"
            if not obs0_fp.exists():
                raise FileNotFoundError(f"Observed centerline not found for current month: {obs0_fp}")
            current_cl = np.asarray(np.load(obs0_fp), dtype=float)
            current_cl = _align_centerline_orientation(ref_orient, current_cl)

        obs1 = None
        obs0_fp = obs_dir / f"{m0}_centerline_geo_1000pts.npy"
        obs1_fp = obs_dir / f"{m1}_centerline_geo_1000pts.npy"
        if str(args.mode) != "forecast":
            if m0 not in month2src:
                raise KeyError(f"Month {m0} not found in dataset months. Check data_dir/tag/target/migration_mode and fold.")
            if not obs1_fp.exists():
                raise FileNotFoundError(f"Observed centerline not found for next month: {obs1_fp}")
            obs1 = np.asarray(np.load(obs1_fp), dtype=float)
            obs1 = _align_centerline_orientation(ref_orient, obs1)

        run_name = f"hc_{m0}_{m1}"
        run_dir = sandbox_root / run_name
        run_input = run_dir / "input"
        run_output = run_dir / "output"
        run_temp = run_dir / "temp"
        run_input.mkdir(parents=True, exist_ok=False)
        run_output.mkdir(parents=True, exist_ok=False)
        run_temp.mkdir(parents=True, exist_ok=False)
        t_io0 = time.perf_counter()

        _, n0 = _write_centerline_xy(
            run_input_dir=run_input,
            coords_utm=current_cl,
            w_ref=w_ref,
            origin_x=origin_x,
            origin_y=origin_y,
            n0_fixed=mcmm_n0_fixed,
        )

        ts_row = None
        ts_row_src = "master"
        srow = None
        if str(args.mode) == "forecast":
            if scenario_map is None:
                raise RuntimeError("scenario_map is missing")
            srow = scenario_map.get(m0)
            if srow is not None:
                ts_row_src = "scenario"
                ts_row = [
                    m0,
                    srow.get("Q", ""),
                    srow.get("W", ""),
                    srow.get("D", ""),
                    srow.get("E_rate", ""),
                    srow.get("physical_ds", ""),
                ]
            else:
                ts_row = ts_map.get(m0)
                if ts_row is None:
                    ts_row = _extract_timeseries_row(master_csv=master_ts, yyyymm=m0)
                srow = _rowlist_to_dict(ts_row)
                ts_row_src = "master"
        else:
            ts_row = ts_map.get(m0)
            if ts_row is None:
                ts_row = _extract_timeseries_row(master_csv=master_ts, yyyymm=m0)
        ts_file = run_input / "timeseries.csv"
        with ts_file.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["month", "Q", "W", "D", "E_rate", "physical_ds"])
            w.writerow(list(ts_row))

        simname = run_name
        sim_file = run_input / "hindcast.sim"
        _make_one_step_sim(
            template_sim=template_sim,
            out_sim=sim_file,
            simname=simname,
            n0=n0,
            timeseries_file=ts_file.name,
            dt0="0.083333333333d0",
        )

        t_io1 = time.perf_counter()

        t_mcmm0 = time.perf_counter()
        _run_mcmm_once(mcmm_exe=mcmm_exe, run_dir=run_dir, sim_filename=sim_file.name)
        t_mcmm1 = time.perf_counter()

        n_conf = len(list(run_output.glob("configuration_*.out")))

        t_parse0 = time.perf_counter()
        if args.parse_mode == "script":
            _run_parse_output(
                parse_script=parse_script,
                run_output_dir=run_output,
                simname=simname,
                start_yyyymm=m0,
                initial_month=start,
            )
            parsed_dir = run_output / "mcmm_centerlines_utm_1000pts"
            manifest = parsed_dir / "mcmm_centerlines_manifest.csv"
            if not manifest.exists():
                raise FileNotFoundError(f"Manifest not found: {manifest}")
            import pandas as pd

            dfm = pd.read_csv(manifest)
            dfm["yyyymm"] = dfm["yyyymm"].astype(str).str.replace(".0", "", regex=False)
            hit = dfm[dfm["yyyymm"] == m1]
            if hit.empty:
                raise RuntimeError(f"Parsed outputs missing target month {m1} in {manifest}")
            src_name = str(hit.iloc[0]["file"])
            mc1_fp = parsed_dir / src_name
            if not mc1_fp.exists():
                raise FileNotFoundError(f"MCMM parsed npy not found: {mc1_fp}")
            mc1 = np.asarray(np.load(mc1_fp), dtype=float)
        else:
            mc1 = _parse_mcmm_centerline_utm(
                run_output_dir=run_output,
                simname=simname,
                start_yyyymm=m0,
                target_yyyymm=m1,
                w_ref=w_ref,
                origin_x=origin_x,
                origin_y=origin_y,
                n_out=1000,
            )
        t_parse1 = time.perf_counter()
        mc1 = _align_centerline_orientation(current_cl, mc1)

        if obs1 is None:
            mcmm_pw = {"pw_mean": float("nan"), "pw_p95": float("nan"), "pw_max": float("nan")}
            mcmm_nn = {"nn_mean": float("nan"), "nn_p95": float("nan"), "nn_max": float("nan")}
        else:
            mc1_eval = _align_centerline_orientation(obs1, mc1)
            mcmm_pw = _pointwise_metrics(mc1_eval, obs1)
            mcmm_nn = _nn_metrics(mc1_eval, obs1)

        mc1_state = mc1
        if args.migration_mode == "min":
            tree = cKDTree(mc1)
            y_mcmm, idx = tree.query(current_cl, k=1)
            mc1_map = mc1[np.asarray(idx, dtype=int)]
            delta_vec = mc1_map - current_cl
            mc1_state = mc1_map
        else:
            delta_vec = mc1 - current_cl
            y_mcmm = np.linalg.norm(delta_vec, axis=1)

        data_obj = None
        sample_idx = None

        data_ratio = None
        data_y_obs = None
        data_y_mcmm = None
        if str(args.mode) != "forecast":
            data_obj, sample_idx = month2src[m0]
            if hasattr(data_obj, "y") and getattr(data_obj, "y") is not None:
                data_ratio = np.asarray(data_obj.y[sample_idx].detach().cpu().numpy(), dtype=float).reshape(-1)
            if hasattr(data_obj, "y_obs") and getattr(data_obj, "y_obs") is not None:
                data_y_obs = np.asarray(data_obj.y_obs[sample_idx].detach().cpu().numpy(), dtype=float).reshape(-1)
            if hasattr(data_obj, "y_mcmm") and getattr(data_obj, "y_mcmm") is not None:
                data_y_mcmm = np.asarray(data_obj.y_mcmm[sample_idx].detach().cpu().numpy(), dtype=float).reshape(-1)

        x = None
        x_width = None
        x_depth = None
        depth_mask_tensor = None
        if not bool(getattr(args, "oracle_use_gt", False)):
            mcmm_mean = float(np.nanmean(y_mcmm)) if np.isfinite(np.nanmean(y_mcmm)) else 0.0
            mcmm_std = float(np.nanstd(y_mcmm)) if np.isfinite(np.nanstd(y_mcmm)) else 1.0
            mcmm_std = mcmm_std if mcmm_std > 1e-6 else 1.0
            x_mcmm = (np.asarray(y_mcmm, dtype=float) - mcmm_mean) / mcmm_std
            x_mcmm = np.nan_to_num(x_mcmm, nan=0.0, posinf=0.0, neginf=0.0)

            if str(args.mode) == "forecast":
                hs = getattr(train_data, "hydro_stats", None)
                if hs is None:
                    hs = {"Q_mean": 0.0, "Q_std": 1.0, "Q_cumsum3_mean": 0.0, "Q_cumsum3_std": 1.0, "Qs_mean": 0.0, "Qs_std": 1.0}
                Q_mean = float(hs.get("Q_mean", 0.0))
                Q_std = float(hs.get("Q_std", 1.0))
                Q_std = Q_std if Q_std > 1e-6 else 1.0
                Qc_mean = float(hs.get("Q_cumsum3_mean", 0.0))
                Qc_std = float(hs.get("Q_cumsum3_std", 1.0))
                Qc_std = Qc_std if Qc_std > 1e-6 else 1.0
                Qs_mean = float(hs.get("Qs_mean", 0.0))
                Qs_std = float(hs.get("Qs_std", 1.0))
                Qs_std = Qs_std if Qs_std > 1e-6 else 1.0

                q = _try_float(srow.get("Q") if srow else None, default=float("nan"))
                if not np.isfinite(q):
                    q = 0.0
                qs = _try_float(srow.get("Qs") if srow else None, default=0.0)
                if not np.isfinite(qs):
                    qs = 0.0

                if not np.isfinite(last_q):
                    q_prev = q
                else:
                    q_prev = float(last_q)

                dQ = (q - q_prev) / (q_prev + 1e-6) if np.isfinite(q_prev) else 0.0
                dQ = float(np.clip(dQ, -2.0, 2.0))

                q_hist2 = list(q_hist)
                q_hist2.append(q)
                q_hist2 = q_hist2[-3:]
                q_cumsum3 = float(np.nansum(np.asarray(q_hist2, dtype=float)))

                y_m0, m_m0 = _parse_yyyymm(m0)
                season_sin = float(np.sin(2 * np.pi * m_m0 / 12.0))
                season_cos = float(np.cos(2 * np.pi * m_m0 / 12.0))

                Q_norm = (q - Q_mean) / Q_std
                Q_prev_norm = (q_prev - Q_mean) / Q_std
                Q_cumsum3_norm = (q_cumsum3 - Qc_mean) / Qc_std
                Qs_norm = (qs - Qs_mean) / Qs_std

                hydro_feat = torch.tensor(
                    [
                        float(Q_norm),
                        float(Q_prev_norm),
                        float(dQ),
                        float(Q_cumsum3_norm),
                        float(Qs_norm),
                        float(season_sin),
                        float(season_cos),
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                hydro_feat = torch.where(torch.isfinite(hydro_feat), hydro_feat, torch.zeros_like(hydro_feat))
                x_hydro = hydro_feat.unsqueeze(0).expand(x_static.shape[0], -1)

                width_mean = _try_float(getattr(train_data, "width_mean", None), default=0.0)
                width_std = _try_float(getattr(train_data, "width_std", None), default=1.0)
                width_std = width_std if np.isfinite(width_std) and width_std > 1e-6 else 1.0
                depth_mean = _try_float(getattr(train_data, "depth_mean", None), default=0.0)
                depth_std = _try_float(getattr(train_data, "depth_std", None), default=1.0)
                depth_std = depth_std if np.isfinite(depth_std) and depth_std > 1e-6 else 1.0

                n_nodes = int(x_static.shape[0])
                if use_width:
                    wvec = _load_along_vector_csv(Path(args.width_along_csv) if args.width_along_csv else None, m0, n_expected=n_nodes)
                    if wvec is None:
                        w_scalar = _try_float(srow.get("W") if srow else None, default=0.0)
                        wvec = np.full((n_nodes,), float(w_scalar), dtype=float)
                    wvec = np.nan_to_num(wvec, nan=width_mean, posinf=width_mean, neginf=width_mean)
                    w_norm = (wvec - width_mean) / width_std
                    w_norm = np.nan_to_num(w_norm, nan=0.0, posinf=0.0, neginf=0.0)
                    x_width = torch.tensor(w_norm, dtype=torch.float32, device=device)

                if use_depth:
                    dvec = _load_along_vector_csv(Path(args.depth_along_csv) if args.depth_along_csv else None, m0, n_expected=n_nodes)
                    if dvec is None:
                        d_scalar = _try_float(srow.get("D") if srow else None, default=0.0)
                        dvec = np.full((n_nodes,), float(d_scalar), dtype=float)
                    d_mask = np.isfinite(dvec).astype(np.float32)
                    dvec = np.nan_to_num(dvec, nan=depth_mean, posinf=depth_mean, neginf=depth_mean)
                    d_norm = (dvec - depth_mean) / depth_std
                    d_norm = np.nan_to_num(d_norm, nan=0.0, posinf=0.0, neginf=0.0)
                    x_depth = torch.tensor(d_norm, dtype=torch.float32, device=device)

                    if bool(model_cfg.get("use_depth_mask", False)):
                        dm = torch.tensor(d_mask, dtype=torch.float32, device=device)
                        dm = torch.where(torch.isfinite(dm), dm, torch.zeros_like(dm))
                        depth_mask_tensor = dm
            else:
                hydro_feat = data_obj.x_hydro[sample_idx].to(device)
                x_hydro = hydro_feat.unsqueeze(0).expand(x_static.shape[0], -1)

            x_parts = [x_static, x_hydro]
            if bool(model_cfg.get("use_mcmm", False)):
                x_parts.append(torch.tensor(x_mcmm, dtype=torch.float32, device=device).unsqueeze(1))
            if bool(model_cfg.get("use_depth_mask", False)):
                if str(args.mode) == "forecast":
                    if depth_mask_tensor is None:
                        depth_mask_tensor = torch.zeros((x_static.shape[0],), dtype=torch.float32, device=device)
                    x_parts.append(depth_mask_tensor.unsqueeze(1).to(dtype=torch.float32))
                else:
                    if not (hasattr(data_obj, "x_depth_mask") and data_obj.x_depth_mask is not None):
                        raise AttributeError(
                            "Checkpoint requires depth_mask feature (use_depth_mask=True), "
                            "but dataset has no x_depth_mask. Please rebuild dataset or retrain without depth_mask."
                        )
                    dm = data_obj.x_depth_mask[sample_idx].to(device)
                    x_parts.append(dm.unsqueeze(1).to(dtype=torch.float32))
            x = torch.cat(x_parts, dim=1)

            if str(args.mode) != "forecast":
                if use_width and hasattr(data_obj, "x_width") and data_obj.x_width is not None:
                    x_width = data_obj.x_width[sample_idx].to(device)
                if use_depth and hasattr(data_obj, "x_depth") and data_obj.x_depth is not None:
                    x_depth = data_obj.x_depth[sample_idx].to(device)

        t_gnn0 = time.perf_counter()
        target_name = str(args.target)
        pred_ratio = np.ones_like(y_mcmm, dtype=float)
        pred_dn = None
        ratio_sat_frac = float("nan")
        if bool(getattr(args, "oracle_use_gt", False)):
            if target_name == "dn":
                if bool(getattr(args, "oracle_dn_online", False)):
                    if args.migration_mode == "min":
                        base = np.asarray(mc1_map, dtype=float)
                        idx_map = np.asarray(idx, dtype=int)
                    else:
                        base = np.asarray(mc1, dtype=float)
                        idx_map = None

                    cl_mc1 = np.asarray(mc1, dtype=float)
                    n_mc1 = int(cl_mc1.shape[0])
                    tang = np.zeros_like(cl_mc1, dtype=float)
                    if n_mc1 >= 2:
                        tang[0] = cl_mc1[1] - cl_mc1[0]
                        tang[-1] = cl_mc1[-1] - cl_mc1[-2]
                        if n_mc1 > 2:
                            tang[1:-1] = cl_mc1[2:] - cl_mc1[:-2]
                    tang_norm = np.linalg.norm(tang, axis=1)
                    ok = tang_norm > 1e-12
                    tang_unit = np.zeros_like(tang, dtype=float)
                    tang_unit[ok] = tang[ok] / tang_norm[ok, None]
                    normal_mc1 = np.stack([-tang_unit[:, 1], tang_unit[:, 0]], axis=1)

                    if idx_map is not None:
                        normal_map = normal_mc1[idx_map]
                    else:
                        normal_map = normal_mc1

                    obs1_dn = _align_centerline_orientation(mc1, obs1)
                    tree_obs = cKDTree(obs1_dn)
                    _, idx_obs = tree_obs.query(base, k=1)
                    obs1_map = obs1_dn[np.asarray(idx_obs, dtype=int)]

                    pred_dn = np.sum((obs1_map - base) * normal_map, axis=1)
                    pred_dn = np.asarray(pred_dn, dtype=float).reshape(-1)
                    pred_dn = float(args.dn_gain) * pred_dn
                    dn_clip = float(getattr(args, "dn_clip", 0.0))
                    if np.isfinite(dn_clip) and dn_clip > 0:
                        pred_dn = np.clip(pred_dn, -dn_clip, dn_clip)
                else:
                    if data_ratio is None:
                        raise ValueError("--oracle_use_gt requires dataset to provide y target (data_obj.y) when --target=dn")
                    pred_dn = np.asarray(data_ratio, dtype=float).reshape(-1)
                    pred_dn = float(args.dn_gain) * pred_dn
                    dn_clip = float(getattr(args, "dn_clip", 0.0))
                    if np.isfinite(dn_clip) and dn_clip > 0:
                        pred_dn = np.clip(pred_dn, -dn_clip, dn_clip)
            else:
                if data_ratio is None:
                    raise ValueError("--oracle_use_gt requires dataset to provide y target (data_obj.y)")
                pred_ratio = np.asarray(data_ratio, dtype=float).reshape(-1)
                pred_ratio = 1.0 + float(args.ratio_gain) * (pred_ratio - 1.0)
                pred_ratio = np.clip(pred_ratio, 0.0, float(args.ratio_max))
            t_gnn1 = time.perf_counter()
        else:
            if target_name == "dn":
                if float(args.dn_gain) == 0.0:
                    pred_dn = np.zeros_like(y_mcmm, dtype=float)
                    t_gnn1 = time.perf_counter()
                else:
                    if args.profile and device.type == "cuda":
                        torch.cuda.synchronize()
                    with torch.no_grad():
                        pred_dn = model(x, edge_index, x_width=x_width, x_depth=x_depth).squeeze().cpu().numpy()
                    if args.profile and device.type == "cuda":
                        torch.cuda.synchronize()
                    t_gnn1 = time.perf_counter()
                    pred_dn = np.asarray(pred_dn, dtype=float).reshape(-1)
                    pred_dn = float(args.dn_gain) * pred_dn
                    dn_clip = float(getattr(args, "dn_clip", 0.0))
                    if np.isfinite(dn_clip) and dn_clip > 0:
                        pred_dn = np.clip(pred_dn, -dn_clip, dn_clip)
            else:
                if float(args.ratio_gain) == 0.0:
                    pred_ratio = np.ones_like(y_mcmm, dtype=float)
                    t_gnn1 = time.perf_counter()
                else:
                    if args.profile and device.type == "cuda":
                        torch.cuda.synchronize()
                    with torch.no_grad():
                        pred_ratio = model(x, edge_index, x_width=x_width, x_depth=x_depth).squeeze().cpu().numpy()
                    if args.profile and device.type == "cuda":
                        torch.cuda.synchronize()
                    t_gnn1 = time.perf_counter()
                    pred_ratio = np.asarray(pred_ratio, dtype=float).reshape(-1)
                    pred_ratio = 1.0 + float(args.ratio_gain) * (pred_ratio - 1.0)
                    pred_ratio = np.clip(pred_ratio, 0.0, float(args.ratio_max))

        if target_name == "ratio":
            ratio_max = float(args.ratio_max)
            if np.isfinite(ratio_max) and ratio_max > 0:
                ratio_sat_frac = float(np.mean(pred_ratio >= (ratio_max - 1e-6)))

        ratio_corr_gt = float("nan")
        ratio_rmse_gt = float("nan")
        ycorr_mcmm_gt = float("nan")
        yrmse_mcmm_gt = float("nan")
        ycorr_pred_gt = float("nan")
        yrmse_pred_gt = float("nan")
        dn_mean = float("nan")
        dn_p95 = float("nan")
        dn_max = float("nan")
        dn_corr_gt = float("nan")
        dn_rmse_gt = float("nan")
        dn_expert = None
        ratio_expert = None
        if data_ratio is not None and len(data_ratio) == len(pred_ratio):
            ratio_corr_gt = _corr(pred_ratio, data_ratio)
            ratio_rmse_gt = float(np.sqrt(np.mean((pred_ratio - data_ratio) ** 2)))
        if data_y_obs is not None and data_y_mcmm is not None and len(data_y_obs) == len(pred_ratio):
            y_base_gt = data_y_mcmm
            y_pred_gt = data_y_mcmm * pred_ratio
            ycorr_mcmm_gt = _corr(y_base_gt, data_y_obs)
            yrmse_mcmm_gt = float(np.sqrt(np.mean((y_base_gt - data_y_obs) ** 2)))
            ycorr_pred_gt = _corr(y_pred_gt, data_y_obs)
            yrmse_pred_gt = float(np.sqrt(np.mean((y_pred_gt - data_y_obs) ** 2)))
        if target_name == "dn" and data_ratio is not None and pred_dn is not None:
            pred_dn = np.asarray(pred_dn, dtype=float).reshape(-1)
            if len(pred_dn) != len(data_ratio):
                pred_dn = np.zeros_like(data_ratio, dtype=float)
            dn_mean = float(np.mean(pred_dn))
            dn_p95 = float(np.percentile(pred_dn, 95))
            dn_max = float(np.max(np.abs(pred_dn)))
            dn_corr_gt = _corr(pred_dn, data_ratio)
            dn_rmse_gt = float(np.sqrt(np.mean((pred_dn - data_ratio) ** 2)))

        if target_name == "dn":
            disp_pred_clip_frac = float("nan")
            if args.migration_mode == "min":
                base = np.asarray(mc1_map, dtype=float)
                idx_map = np.asarray(idx, dtype=int)
            else:
                base = np.asarray(mc1, dtype=float)
                idx_map = None

            cl_mc1 = np.asarray(mc1, dtype=float)
            n_mc1 = int(cl_mc1.shape[0])
            tang = np.zeros_like(cl_mc1, dtype=float)
            if n_mc1 >= 2:
                tang[0] = cl_mc1[1] - cl_mc1[0]
                tang[-1] = cl_mc1[-1] - cl_mc1[-2]
                if n_mc1 > 2:
                    tang[1:-1] = cl_mc1[2:] - cl_mc1[:-2]
            tang_norm = np.linalg.norm(tang, axis=1)
            ok = tang_norm > 1e-12
            tang_unit = np.zeros_like(tang, dtype=float)
            tang_unit[ok] = tang[ok] / tang_norm[ok, None]
            normal_mc1 = np.stack([-tang_unit[:, 1], tang_unit[:, 0]], axis=1)

            if idx_map is not None:
                normal_map = normal_mc1[idx_map]
            else:
                normal_map = normal_mc1

            if dagger_dump_dir is not None:
                obs1_dn = _align_centerline_orientation(mc1, obs1)
                tree_obs = cKDTree(obs1_dn)
                _, idx_obs = tree_obs.query(base, k=1)
                obs1_map = obs1_dn[np.asarray(idx_obs, dtype=int)]
                dn_expert = np.sum((obs1_map - base) * normal_map, axis=1)
                dn_expert = np.asarray(dn_expert, dtype=float).reshape(-1)
                dn_clip = float(getattr(args, "dn_clip", 0.0))
                if np.isfinite(dn_clip) and dn_clip > 0:
                    dn_expert = np.clip(dn_expert, -dn_clip, dn_clip)

            pred_dn_eff = np.zeros(base.shape[0], dtype=float)
            if pred_dn is not None:
                pred_dn_eff = np.asarray(pred_dn, dtype=float).reshape(-1)
                if len(pred_dn_eff) != base.shape[0]:
                    pred_dn_eff = np.zeros(base.shape[0], dtype=float)
            pred_dn_eff = np.nan_to_num(pred_dn_eff, nan=0.0, posinf=0.0, neginf=0.0)

            if normal_map.shape != base.shape:
                next_pred = base
            else:
                next_pred = base + pred_dn_eff[:, None] * normal_map

            disp_vec = np.asarray(next_pred, dtype=float) - np.asarray(current_cl, dtype=float)
            disp_norm = np.linalg.norm(disp_vec, axis=1)
            max_disp = float(args.max_node_disp_m)
            if np.isfinite(max_disp) and max_disp > 0:
                scale = np.ones_like(disp_norm)
                mask = disp_norm > max_disp
                scale[mask] = max_disp / (disp_norm[mask] + 1e-12)
                disp_vec = disp_vec * scale[:, None]
                next_pred = np.asarray(current_cl, dtype=float) + disp_vec

                disp_pred_clip_frac = float(np.mean(mask)) if disp_norm.size else float("nan")
            elif disp_norm.size:
                disp_pred_clip_frac = 0.0
        else:
            disp_pred_clip_frac = float("nan")
            if dagger_dump_dir is not None:
                query_pts = np.asarray((mc1_map if args.migration_mode == "min" else mc1), dtype=float)
                obs1_ratio = _align_centerline_orientation(query_pts, obs1)
                tree_obs = cKDTree(obs1_ratio)
                _, idx_obs = tree_obs.query(query_pts, k=1)
                obs1_map = obs1_ratio[np.asarray(idx_obs, dtype=int)]

                disp_obs = obs1_map - np.asarray(current_cl, dtype=float)
                delta_eff = np.asarray(delta_vec, dtype=float)
                if str(getattr(args, "disp_project", "none")) == "normal":
                    delta_eff = _project_disp_to_normal(current_cl, delta_eff)

                denom = np.sum(delta_eff * delta_eff, axis=1)
                ratio_expert = np.ones(denom.shape[0], dtype=float)
                ok = denom > 1e-12
                if np.any(ok):
                    ratio_expert[ok] = np.sum(disp_obs[ok] * delta_eff[ok], axis=1) / denom[ok]
                ratio_expert = np.nan_to_num(ratio_expert, nan=1.0, posinf=1.0, neginf=1.0)
                ratio_expert = np.clip(ratio_expert, 0.0, float(args.ratio_max))

            if float(args.ratio_gain) == 0.0:
                next_pred = mc1_state
                disp_pred_clip_frac = 0.0
            else:
                disp_vec = pred_ratio[:, None] * delta_vec

                if str(getattr(args, "disp_project", "none")) == "normal":
                    disp_vec = _project_disp_to_normal(current_cl, disp_vec)

                disp_norm = np.linalg.norm(disp_vec, axis=1)
                max_disp = float(args.max_node_disp_m)
                if np.isfinite(max_disp) and max_disp > 0:
                    scale = np.ones_like(disp_norm)
                    mask = disp_norm > max_disp
                    scale[mask] = max_disp / (disp_norm[mask] + 1e-12)
                    disp_vec = disp_vec * scale[:, None]

                    disp_pred_clip_frac = float(np.mean(mask)) if disp_norm.size else float("nan")
                elif disp_norm.size:
                    disp_pred_clip_frac = 0.0

                next_pred = current_cl + disp_vec

        disp_mcmm_vec = np.asarray(delta_vec, dtype=float)
        if disp_mcmm_vec.shape != np.asarray(current_cl, dtype=float).shape:
            disp_mcmm_vec = np.asarray(mc1, dtype=float) - np.asarray(current_cl, dtype=float)
        if str(getattr(args, "disp_project", "none")) == "normal":
            disp_mcmm_vec = _project_disp_to_normal(current_cl, disp_mcmm_vec)

        disp_mcmm_norm = np.linalg.norm(disp_mcmm_vec, axis=1)
        disp_mcmm_norm = np.nan_to_num(disp_mcmm_norm, nan=0.0, posinf=0.0, neginf=0.0)

        disp_pred_norm = np.linalg.norm(np.asarray(next_pred, dtype=float) - np.asarray(current_cl, dtype=float), axis=1)
        disp_pred_norm = np.nan_to_num(disp_pred_norm, nan=0.0, posinf=0.0, neginf=0.0)

        if dagger_dump_dir is not None:
            if str(args.mode) != "rolling":
                raise ValueError("--dagger_dump_dir is only supported in --mode rolling")
            if str(args.target) not in ["dn", "ratio"]:
                raise ValueError("--dagger_dump_dir currently supports only --target dn or ratio")
            if x is None:
                raise ValueError("--dagger_dump_dir requires model features x (do not use --oracle_use_gt)")

            target_name = str(args.target)
            if target_name == "dn":
                y_expert = dn_expert
            elif target_name == "ratio":
                y_expert = ratio_expert
            else:
                y_expert = None
            if y_expert is None:
                raise ValueError(f"Failed to compute expert target for DAgger dump: {target_name}")

            sample = {
                "m0": str(m0),
                "m1": str(m1),
                "fold": int(fold),
                "migration_mode": str(args.migration_mode),
                "use_width": bool(use_width),
                "use_depth": bool(use_depth),
                "x": x.detach().cpu().to(torch.float32),
                "edge_index": train_data.edge_index.detach().cpu(),
                "x_width": (x_width.detach().cpu().to(torch.float32) if x_width is not None else None),
                "x_depth": (x_depth.detach().cpu().to(torch.float32) if x_depth is not None else None),
                "target": str(target_name),
                "y": torch.tensor(y_expert, dtype=torch.float32),
            }
            sample_fp = dagger_dump_dir / f"sample_{k:02d}_{m0}_{m1}.pt"
            torch.save(sample, sample_fp)

        np.save(pred_dir / f"{m1}_centerline_geo_1000pts.npy", next_pred)

        t_metric0 = time.perf_counter()
        if obs1 is None:
            pw = {"pw_mean": float("nan"), "pw_p95": float("nan"), "pw_max": float("nan")}
            nn = {"nn_mean": float("nan"), "nn_p95": float("nan"), "nn_max": float("nan")}
        else:
            next_eval = _align_centerline_orientation(obs1, next_pred)
            pw = _pointwise_metrics(next_eval, obs1)
            nn = _nn_metrics(next_eval, obs1)

        obs0 = np.asarray(np.load(obs0_fp), dtype=float) if (obs0_fp is not None and obs0_fp.exists()) else None

        mig_corr = float("nan")
        mig_rmse = float("nan")
        mcmm_mig_corr = float("nan")
        mcmm_mig_rmse = float("nan")
        mcmm_field_corr = float("nan")
        mcmm_field_rmse = float("nan")
        pred_field_corr = float("nan")
        pred_field_rmse = float("nan")
        if args.mode == "teacher" and obs1 is not None and obs0 is not None:
            mig_obs = _compute_migration(obs0, obs1, mode=args.migration_mode)
            mcmm_mig = _compute_migration(current_cl, mc1, mode=args.migration_mode)
            mig_pred = _compute_migration(current_cl, next_pred, mode=args.migration_mode)

            y_base = np.asarray(y_mcmm, dtype=float)
            y_pred = np.asarray(y_mcmm, dtype=float) * np.asarray(pred_ratio, dtype=float)
            mcmm_field_corr = _corr(y_base, mig_obs)
            mcmm_field_rmse = float(np.sqrt(np.mean((y_base - np.asarray(mig_obs)) ** 2)))
            pred_field_corr = _corr(y_pred, mig_obs)
            pred_field_rmse = float(np.sqrt(np.mean((y_pred - np.asarray(mig_obs)) ** 2)))

            mcmm_mig_corr = _corr(mcmm_mig, mig_obs)
            mcmm_mig_rmse = float(np.sqrt(np.mean((np.asarray(mcmm_mig) - np.asarray(mig_obs)) ** 2)))
            mig_corr = _corr(mig_pred, mig_obs)
            mig_rmse = float(np.sqrt(np.mean((np.asarray(mig_pred) - np.asarray(mig_obs)) ** 2)))
        t_metric1 = time.perf_counter()

        rec = {
            "m0": m0,
            "m1": m1,
            "n0": int(n0),
            "n_conf": int(n_conf),
            "ts_src": str(ts_row_src),
            "dn_mean": float(dn_mean),
            "dn_p95": float(dn_p95),
            "dn_max": float(dn_max),
            "dn_corr_gt": float(dn_corr_gt),
            "dn_rmse_gt": float(dn_rmse_gt),
            "ratio_mean": float(np.mean(pred_ratio)),
            "ratio_p95": float(np.percentile(pred_ratio, 95)),
            "ratio_max": float(np.max(pred_ratio)),
            "ratio_sat_frac": float(ratio_sat_frac),
            "ratio_corr_gt": float(ratio_corr_gt),
            "ratio_rmse_gt": float(ratio_rmse_gt),
            "y_corr_mcmm_gt": float(ycorr_mcmm_gt),
            "y_rmse_mcmm_gt": float(yrmse_mcmm_gt),
            "y_corr_pred_gt": float(ycorr_pred_gt),
            "y_rmse_pred_gt": float(yrmse_pred_gt),
            "disp_mcmm_mean": float(np.mean(disp_mcmm_norm)) if disp_mcmm_norm.size else float("nan"),
            "disp_mcmm_p95": float(np.percentile(disp_mcmm_norm, 95)) if disp_mcmm_norm.size else float("nan"),
            "disp_mcmm_max": float(np.max(disp_mcmm_norm)) if disp_mcmm_norm.size else float("nan"),
            "disp_pred_mean": float(np.mean(disp_pred_norm)) if disp_pred_norm.size else float("nan"),
            "disp_pred_p95": float(np.percentile(disp_pred_norm, 95)) if disp_pred_norm.size else float("nan"),
            "disp_pred_max": float(np.max(disp_pred_norm)) if disp_pred_norm.size else float("nan"),
            "disp_pred_clip_frac": float(disp_pred_clip_frac),
            "disp_roll_mean": float("nan"),
            "disp_roll_p95": float("nan"),
            "disp_roll_max": float("nan"),
            "disp_roll_clip_frac": float("nan"),
            "mcmm_pw_mean": float(mcmm_pw["pw_mean"]),
            "mcmm_pw_p95": float(mcmm_pw["pw_p95"]),
            "mcmm_pw_max": float(mcmm_pw["pw_max"]),
            "mcmm_nn_mean": float(mcmm_nn["nn_mean"]),
            "mcmm_nn_p95": float(mcmm_nn["nn_p95"]),
            "mcmm_nn_max": float(mcmm_nn["nn_max"]),
            **pw,
            **nn,
            "mcmm_mig_corr": float(mcmm_mig_corr),
            "mcmm_mig_rmse": float(mcmm_mig_rmse),
            "mcmm_field_corr": float(mcmm_field_corr),
            "mcmm_field_rmse": float(mcmm_field_rmse),
            "pred_field_corr": float(pred_field_corr),
            "pred_field_rmse": float(pred_field_rmse),
            "mig_corr": float(mig_corr),
            "mig_rmse": float(mig_rmse),
            "redline_level": 0,
            "redline_action": "",
            "redline_update": str(args.rolling_update),
            "redline_alpha": float(args.rolling_blend_alpha),
            "redline_l1_streak": int(redline_l1_streak),
            "redline_l2_streak": int(redline_l2_streak),
        }

        if args.mode in ["rolling", "forecast"]:
            update_eff = str(args.rolling_update)
            alpha_eff = float(args.rolling_blend_alpha)
            redline_level = 0
            redline_action = ""

            if bool(getattr(args, "enable_redline", False)):
                cons = int(getattr(args, "redline_consecutive", 2))
                cons = max(1, cons)

                rs = float(ratio_sat_frac) if np.isfinite(ratio_sat_frac) else 0.0
                pc = float(disp_pred_clip_frac) if np.isfinite(disp_pred_clip_frac) else 0.0

                max_disp = float(args.max_node_disp_m)

                a0 = float(np.clip(alpha_eff, 0.0, 1.0))
                disp_mcmm0 = np.asarray(delta_vec, dtype=float)
                disp_pred0 = np.asarray(next_pred, dtype=float) - np.asarray(current_cl, dtype=float)
                if update_eff == "mcmm":
                    disp_roll0 = np.asarray(mc1_state, dtype=float) - np.asarray(current_cl, dtype=float)
                elif update_eff == "blend":
                    disp_roll0 = (1.0 - a0) * disp_mcmm0 + a0 * disp_pred0
                else:
                    disp_roll0 = disp_pred0

                roll_norm0 = np.linalg.norm(np.asarray(disp_roll0, dtype=float), axis=1)
                roll_norm0 = np.nan_to_num(roll_norm0, nan=0.0, posinf=0.0, neginf=0.0)
                if np.isfinite(max_disp) and max_disp > 0 and roll_norm0.size:
                    rc0 = float(np.mean(roll_norm0 > max_disp))
                else:
                    rc0 = 0.0

                if (np.isfinite(float(getattr(args, "redline_l2_roll_clip_instant", 0.20))) and rc0 > float(getattr(args, "redline_l2_roll_clip_instant", 0.20))):
                    redline_level = 2
                    redline_action = "fallback_mcmm"
                    update_eff = "mcmm"
                else:
                    l2_now = (rs > float(getattr(args, "redline_l2_ratio_sat", 0.35))) or (pc > float(getattr(args, "redline_l2_clip_frac", 0.10))) or (rc0 > float(getattr(args, "redline_l2_clip_frac", 0.10)))
                    l1_now = (rs > float(getattr(args, "redline_l1_ratio_sat", 0.25))) or (pc > float(getattr(args, "redline_l1_clip_frac", 0.05))) or (rc0 > float(getattr(args, "redline_l1_clip_frac", 0.05)))

                    if l2_now:
                        redline_l2_streak += 1
                    else:
                        redline_l2_streak = 0

                    if l1_now:
                        redline_l1_streak += 1
                    else:
                        redline_l1_streak = 0

                    l0_now = (rs > float(getattr(args, "redline_l0_ratio_sat", 0.20))) or (pc > float(getattr(args, "redline_l0_clip_frac", 0.03))) or (rc0 > float(getattr(args, "redline_l0_clip_frac", 0.03)))
                    if l0_now:
                        redline_level = 0
                        redline_action = "watch"

                    if redline_l2_streak >= cons:
                        redline_level = 2
                        redline_action = "fallback_mcmm"
                        update_eff = "mcmm"
                    elif redline_l1_streak >= cons:
                        redline_level = 1
                        update_eff = "blend"
                        alpha_eff = float(getattr(args, "redline_alpha_reduced", 0.2))
                        alpha_eff = float(np.clip(alpha_eff, 0.0, 1.0))
                        redline_action = f"alpha->{alpha_eff:.3f}"

            rec["redline_level"] = int(redline_level)
            rec["redline_action"] = str(redline_action)
            rec["redline_update"] = str(update_eff)
            rec["redline_alpha"] = float(alpha_eff)
            rec["redline_l1_streak"] = int(redline_l1_streak)
            rec["redline_l2_streak"] = int(redline_l2_streak)

            current_cl_old = np.asarray(current_cl, dtype=float)
            if str(update_eff) == "mcmm":
                current_cl = mc1_state
            elif str(update_eff) == "blend":
                a = float(alpha_eff)
                if not np.isfinite(a):
                    a = 0.5
                a = float(np.clip(a, 0.0, 1.0))
                current_cl_old = np.asarray(current_cl, dtype=float)
                disp_mcmm = np.asarray(delta_vec, dtype=float)
                disp_pred = np.asarray(next_pred, dtype=float) - current_cl_old
                disp_blend = (1.0 - a) * disp_mcmm + a * disp_pred
                current_cl = current_cl_old + disp_blend
                current_cl = _align_centerline_orientation(ref_orient, current_cl)
            else:
                current_cl = next_pred

            disp_roll_norm = np.linalg.norm(np.asarray(current_cl, dtype=float) - np.asarray(current_cl_old, dtype=float), axis=1)
            disp_roll_norm = np.nan_to_num(disp_roll_norm, nan=0.0, posinf=0.0, neginf=0.0)
            rec["disp_roll_mean"] = float(np.mean(disp_roll_norm)) if disp_roll_norm.size else float("nan")
            rec["disp_roll_p95"] = float(np.percentile(disp_roll_norm, 95)) if disp_roll_norm.size else float("nan")
            rec["disp_roll_max"] = float(np.max(disp_roll_norm)) if disp_roll_norm.size else float("nan")

            max_disp = float(args.max_node_disp_m)
            if np.isfinite(max_disp) and max_disp > 0 and disp_roll_norm.size:
                rec["disp_roll_clip_frac"] = float(np.mean(disp_roll_norm > max_disp))
            elif disp_roll_norm.size:
                rec["disp_roll_clip_frac"] = 0.0

        records.append(rec)

        if str(args.mode) == "forecast":
            q = _try_float(srow.get("Q") if srow else None, default=float("nan"))
            if not np.isfinite(q):
                q = 0.0
            last_q = float(q)
            q_hist.append(float(q))
            q_hist = q_hist[-2:]

        if not args.keep_sandbox:
            shutil.rmtree(run_dir)

        if args.profile:
            t_step1 = time.perf_counter()
            print(
                f"[{k+1:02d}/{steps}] {m0}->{m1} "
                f"n0={int(n0)} "
                f"n_conf={int(n_conf)} "
                f"mcmm_nn_mean={mcmm_nn['nn_mean']:.1f} "
                f"nn_mean={nn['nn_mean']:.1f} "
                f"t_io={t_io1-t_io0:.2f}s "
                f"t_mcmm={t_mcmm1-t_mcmm0:.2f}s "
                f"t_parse={t_parse1-t_parse0:.2f}s "
                f"t_gnn={t_gnn1-t_gnn0:.2f}s "
                f"t_step={t_step1-t_step0:.2f}s"
            )

    csv_name = "hindcast_summary.csv" if str(args.mode) != "forecast" else "forecast_summary.csv"
    csv_path = out_dir / csv_name
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else [])
        w.writeheader()
        for r in records:
            w.writerow(r)

    json_name = "hindcast_summary.json" if str(args.mode) != "forecast" else "forecast_summary.json"
    json_path = out_dir / json_name

    def _safe_mean(key: str) -> float:
        if not records:
            return float("nan")
        vals = []
        for r in records:
            v = r.get(key, float("nan"))
            try:
                v = float(v)
            except Exception:
                v = float("nan")
            if np.isfinite(v):
                vals.append(v)
        if not vals:
            return float("nan")
        return float(np.mean(vals))

    out = {
        "start": start,
        "steps": steps,
        "fold": fold,
        "data_dir": str(data_dir),
        "tag": args.tag,
        "target": args.target,
        "migration_mode": args.migration_mode,
        "mode": args.mode,
        "oracle_use_gt": bool(getattr(args, "oracle_use_gt", False)),
        "oracle_dn_online": bool(getattr(args, "oracle_dn_online", False)),
        "scenario_timeseries": str(scenario_csv) if scenario_csv is not None else None,
        "init_centerline_fp": str(init_fp) if init_fp is not None else None,
        "width_along_csv": str(getattr(args, "width_along_csv", None)) if getattr(args, "width_along_csv", None) else None,
        "depth_along_csv": str(getattr(args, "depth_along_csv", None)) if getattr(args, "depth_along_csv", None) else None,
        "dagger_dump_dir": str(dagger_dump_dir) if dagger_dump_dir is not None else None,
        "rolling_update": str(args.rolling_update),
        "rolling_blend_alpha": float(args.rolling_blend_alpha),
        "enable_redline": bool(getattr(args, "enable_redline", False)),
        "redline_consecutive": int(getattr(args, "redline_consecutive", 2)),
        "redline_alpha_reduced": float(getattr(args, "redline_alpha_reduced", 0.2)),
        "redline_l0_ratio_sat": float(getattr(args, "redline_l0_ratio_sat", 0.20)),
        "redline_l0_clip_frac": float(getattr(args, "redline_l0_clip_frac", 0.03)),
        "redline_l1_ratio_sat": float(getattr(args, "redline_l1_ratio_sat", 0.25)),
        "redline_l1_clip_frac": float(getattr(args, "redline_l1_clip_frac", 0.05)),
        "redline_l2_ratio_sat": float(getattr(args, "redline_l2_ratio_sat", 0.35)),
        "redline_l2_clip_frac": float(getattr(args, "redline_l2_clip_frac", 0.10)),
        "redline_l2_roll_clip_instant": float(getattr(args, "redline_l2_roll_clip_instant", 0.20)),
        "disp_project": str(getattr(args, "disp_project", "none")),
        "mcmm_n0": int(getattr(args, "mcmm_n0", 0)),
        "lock_mcmm_n0": bool(getattr(args, "lock_mcmm_n0", False)),
        "mcmm_n0_fixed": int(mcmm_n0_fixed) if mcmm_n0_fixed is not None else None,
        "use_width": use_width,
        "use_depth": use_depth,
        "model_dir": str(model_dir),
        "ratio_max": float(args.ratio_max),
        "ratio_gain": float(args.ratio_gain),
        "dn_gain": float(getattr(args, "dn_gain", 1.0)),
        "dn_clip": float(getattr(args, "dn_clip", 0.0)),
        "max_node_disp_m": float(args.max_node_disp_m),
        "model_cfg": model_cfg,
        "mean_mcmm_pw_mean": float(np.mean([r["mcmm_pw_mean"] for r in records])) if records else float("nan"),
        "mean_mcmm_nn_mean": float(np.mean([r["mcmm_nn_mean"] for r in records])) if records else float("nan"),
        "mean_pw_mean": float(np.mean([r["pw_mean"] for r in records])) if records else float("nan"),
        "mean_nn_mean": float(np.mean([r["nn_mean"] for r in records])) if records else float("nan"),
        "mean_dn_mean": _safe_mean("dn_mean") if str(args.target) == "dn" else float("nan"),
        "mean_dn_rmse_gt": _safe_mean("dn_rmse_gt") if str(args.target) == "dn" else float("nan"),
        "mean_ratio_corr_gt": _safe_mean("ratio_corr_gt"),
        "mean_ratio_sat_frac": _safe_mean("ratio_sat_frac") if str(args.target) == "ratio" else float("nan"),
        "max_redline_level": int(np.max([int(r.get("redline_level", 0) or 0) for r in records])) if records else 0,
        "count_redline_l1": int(np.sum([1 for r in records if int(r.get("redline_level", 0) or 0) == 1])) if records else 0,
        "count_redline_l2": int(np.sum([1 for r in records if int(r.get("redline_level", 0) or 0) == 2])) if records else 0,
        "mean_y_corr_mcmm_gt": _safe_mean("y_corr_mcmm_gt"),
        "mean_y_corr_pred_gt": _safe_mean("y_corr_pred_gt"),
        "mean_disp_mcmm_mean": _safe_mean("disp_mcmm_mean"),
        "mean_disp_pred_mean": _safe_mean("disp_pred_mean"),
        "mean_disp_pred_clip_frac": _safe_mean("disp_pred_clip_frac"),
        "mean_disp_roll_mean": _safe_mean("disp_roll_mean"),
        "mean_disp_roll_clip_frac": _safe_mean("disp_roll_clip_frac"),
        "mean_mcmm_mig_corr": _safe_mean("mcmm_mig_corr"),
        "mean_mcmm_mig_rmse": _safe_mean("mcmm_mig_rmse"),
        "mean_mcmm_field_corr": _safe_mean("mcmm_field_corr"),
        "mean_mcmm_field_rmse": _safe_mean("mcmm_field_rmse"),
        "mean_pred_field_corr": _safe_mean("pred_field_corr"),
        "mean_pred_field_rmse": _safe_mean("pred_field_rmse"),
        "mean_mig_corr": _safe_mean("mig_corr"),
        "records": records,
    }
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.keep_sandbox:
        shutil.rmtree(sandbox_root, ignore_errors=True)

    if args.profile:
        t_global1 = time.perf_counter()
        print(f"Done. Wall time: {t_global1 - t_global0:.2f}s")


if __name__ == "__main__":
    main()
