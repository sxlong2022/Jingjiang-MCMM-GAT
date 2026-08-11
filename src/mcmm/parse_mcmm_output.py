import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np


def _win_long_path(p: Path) -> str:
    s = str(p)
    if not sys.platform.lower().startswith("win"):
        return s
    if s.startswith("\\\\?\\"):
        return s
    try:
        s = str(Path(s).resolve())
    except Exception:
        s = str(p)
    if s.startswith("\\\\"):
        return "\\\\?\\UNC\\" + s.lstrip("\\")
    return "\\\\?\\" + s


def _parse_simulation_table(simulation_dat: Path):
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
            time = float(parts[2])
            n = int(parts[3])
        except ValueError:
            continue
        rows.append({"jp": jp, "jt": jt, "time": time, "n": n})

    return rows


def _add_months(yyyymm: str, months: int) -> str:
    year = int(yyyymm[:4])
    month = int(yyyymm[4:6])
    idx0 = year * 12 + (month - 1)
    idx = idx0 + months
    y = idx // 12
    m = (idx % 12) + 1
    return f"{y:04d}{m:02d}"


def _load_transform(input_dir: Path, fallback_initial_month: str):
    meta_file = input_dir / "jingjiang_centerline_transform.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        return float(meta["w_ref"]), float(meta["origin_utm_x"]), float(meta["origin_utm_y"])

    codes_dir = input_dir.parent.parent
    candidates = [
        codes_dir / "data" / "raw" / "monthly_centerlines"
    ]
    npy_file = None
    for d in candidates:
        fp = d / f"{fallback_initial_month}_centerline_geo_1000pts.npy"
        if fp.exists():
            npy_file = fp
            break
    if npy_file is None:
        raise FileNotFoundError(
            f"Cannot find reference centerline npy for {fallback_initial_month} under candidates: {candidates}"
        )
    coords = np.load(npy_file)
    coords = np.asarray(coords, dtype=float)
    return float(880.0), float(coords[0, 0]), float(coords[0, 1])


def _read_configuration_xy_nd(path: Path):
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

    d = np.sqrt(np.sum((xy[1:] - xy[:-1]) ** 2, axis=1))
    keep = np.concatenate(([True], d > 1e-12))
    xy = xy[keep]

    if len(xy) < 2:
        raise ValueError(f"Too few non-duplicate points in {path}")

    return xy


def _resample_by_arclength(xy: np.ndarray, n_out: int):
    d = np.sqrt(np.sum((xy[1:] - xy[:-1]) ** 2, axis=1))
    s = np.concatenate(([0.0], np.cumsum(d)))
    total = float(s[-1])
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid arclength")

    target_s = np.linspace(0.0, total, n_out)
    x = np.interp(target_s, s, xy[:, 0])
    y = np.interp(target_s, s, xy[:, 1])
    return np.column_stack([x, y])


def _mean_min_dist(a: np.ndarray, b: np.ndarray):
    diff = a[:, None, :] - b[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    min_d = np.sqrt(np.min(d2, axis=1))
    return float(min_d.mean()), float(min_d.max())


def _qc_against_reference(mcmm_xy_utm: np.ndarray, reference_xy_utm: np.ndarray, n_out: int):
    ref_u = _resample_by_arclength(reference_xy_utm, n_out)
    mcmm_u = _resample_by_arclength(mcmm_xy_utm, n_out)

    d_pw = np.sqrt(np.sum((mcmm_u - ref_u) ** 2, axis=1))
    d_pw_rev = np.sqrt(np.sum((mcmm_u[::-1] - ref_u) ** 2, axis=1))

    pw_mean = float(d_pw.mean())
    pw_p95 = float(np.quantile(d_pw, 0.95))
    pw_max = float(d_pw.max())

    pw_mean_rev = float(d_pw_rev.mean())
    pw_p95_rev = float(np.quantile(d_pw_rev, 0.95))
    pw_max_rev = float(d_pw_rev.max())

    a2b_mean, a2b_max = _mean_min_dist(mcmm_u, ref_u)
    b2a_mean, b2a_max = _mean_min_dist(ref_u, mcmm_u)
    return {
        "pw": {"mean": pw_mean, "p95": pw_p95, "max": pw_max},
        "pw_rev": {"mean": pw_mean_rev, "p95": pw_p95_rev, "max": pw_max_rev},
        "nn": {
            "a2b_mean": a2b_mean,
            "a2b_max": a2b_max,
            "b2a_mean": b2a_mean,
            "b2a_max": b2a_max,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_yyyymm", default="201601")
    parser.add_argument("--end_yyyymm", default=None)
    parser.add_argument("--initial_month", default="201601")
    parser.add_argument("--simname", default="my_jingjiang")
    parser.add_argument("--output_dir", default=None, help="Override MCMM output directory (default: <mcmm_root>/output)")
    parser.add_argument("--n_out", type=int, default=1000)
    parser.add_argument("--output_subdir", default="mcmm_centerlines_utm_1000pts")
    parser.add_argument("--qc_reference", action="store_true")
    parser.add_argument("--orient_to_reference", action="store_true")
    parser.add_argument("--w_ref", type=float, default=None)
    parser.add_argument("--origin_utm_x", type=float, default=None)
    parser.add_argument("--origin_utm_y", type=float, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    input_dir = root
    output_dir = Path(args.output_dir) if args.output_dir else (root / "output")

    if args.w_ref is not None and args.origin_utm_x is not None and args.origin_utm_y is not None:
        w_ref, origin_x, origin_y = float(args.w_ref), float(args.origin_utm_x), float(args.origin_utm_y)
    else:
        w_ref, origin_x, origin_y = _load_transform(input_dir=input_dir, fallback_initial_month=args.initial_month)

    simname = str(args.simname).strip() if args.simname is not None else "my_jingjiang"
    sim_rows = _parse_simulation_table(output_dir / f"{simname}_simulation.dat")
    sim_by_jp = {r["jp"]: r for r in sim_rows}

    config_files = []
    for p in output_dir.glob("configuration_*.out"):
        m = re.match(r"configuration_(\d+)\.out$", p.name)
        if not m:
            continue
        config_files.append((int(m.group(1)), p))
    config_files.sort(key=lambda t: t[0])

    out_dir = output_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_xy = None
    reference_file = None
    if args.qc_reference or args.orient_to_reference:
        codes_dir = input_dir.parent.parent
        candidates = [
            codes_dir / "data" / "raw" / "monthly_centerlines"
        ]
        for d in candidates:
            fp = d / f"{args.initial_month}_centerline_geo_1000pts.npy"
            if fp.exists():
                reference_file = fp
                reference_xy = np.asarray(np.load(reference_file), dtype=float)
                break

    manifest_path = out_dir / "mcmm_centerlines_manifest.csv"
    first_saved = None
    seen_yyyymm = set()
    with open(_win_long_path(manifest_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "jp",
                "jt",
                "time",
                "yyyymm",
                "n_in",
                "n_out",
                "w_ref",
                "origin_utm_x",
                "origin_utm_y",
                "file",
            ],
        )
        writer.writeheader()

        for jp, path in config_files:
            sim = sim_by_jp.get(jp, {})
            time = float(sim.get("time", np.nan))
            jt = int(sim.get("jt", -1)) if "jt" in sim else -1

            yyyymm = ""
            if np.isfinite(time):
                months = int(round(time * 12.0))
                yyyymm = _add_months(args.start_yyyymm, months)

            if yyyymm:
                if args.end_yyyymm and str(yyyymm) > str(args.end_yyyymm):
                    continue
                if yyyymm in seen_yyyymm:
                    continue
                seen_yyyymm.add(yyyymm)

            xy_nd = _read_configuration_xy_nd(path)
            xy_nd_1000 = _resample_by_arclength(xy_nd, args.n_out)

            xy_utm = np.empty_like(xy_nd_1000)
            xy_utm[:, 0] = xy_nd_1000[:, 0] * w_ref + origin_x
            xy_utm[:, 1] = xy_nd_1000[:, 1] * w_ref + origin_y

            if args.orient_to_reference and reference_xy is not None:
                d_same = float(np.linalg.norm(xy_utm[0] - reference_xy[0]) + np.linalg.norm(xy_utm[-1] - reference_xy[-1]))
                d_rev = float(np.linalg.norm(xy_utm[0] - reference_xy[-1]) + np.linalg.norm(xy_utm[-1] - reference_xy[0]))
                if d_rev < d_same:
                    xy_utm = xy_utm[::-1]

            stem = f"configuration_{jp:03d}"
            if yyyymm:
                stem = f"{yyyymm}_configuration_{jp:03d}"

            out_file = out_dir / f"{stem}_centerline_geo_1000pts.npy"
            if out_file.exists():
                out_file = out_dir / f"{stem}_centerline_geo_1000pts_dup.npy"

            np.save(_win_long_path(out_file), xy_utm)

            if first_saved is None:
                first_saved = xy_utm

            writer.writerow(
                {
                    "jp": jp,
                    "jt": jt,
                    "time": time,
                    "yyyymm": yyyymm,
                    "n_in": int(xy_nd.shape[0]),
                    "n_out": int(args.n_out),
                    "w_ref": w_ref,
                    "origin_utm_x": origin_x,
                    "origin_utm_y": origin_y,
                    "file": str(out_file.name),
                }
            )

    print(f"Parsed {len(config_files)} configuration files into: {out_dir}")

    if args.qc_reference and first_saved is not None and reference_xy is not None and reference_file is not None:
        qc = _qc_against_reference(first_saved, reference_xy, args.n_out)
        print(f"QC vs reference {reference_file.name}:")
        print(
            f"  pointwise(mean/p95/max)     = {qc['pw']['mean']:.3f} / {qc['pw']['p95']:.3f} / {qc['pw']['max']:.3f} m"
        )
        print(
            f"  pointwise_rev(mean/p95/max) = {qc['pw_rev']['mean']:.3f} / {qc['pw_rev']['p95']:.3f} / {qc['pw_rev']['max']:.3f} m"
        )
        print(
            f"  nn a->b(mean/max)           = {qc['nn']['a2b_mean']:.3f} / {qc['nn']['a2b_max']:.3f} m"
        )
        print(
            f"  nn b->a(mean/max)           = {qc['nn']['b2a_mean']:.3f} / {qc['nn']['b2a_max']:.3f} m"
        )


if __name__ == "__main__":
    main()
