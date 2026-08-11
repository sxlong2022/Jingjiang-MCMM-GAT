import argparse
import subprocess
import sys
from pathlib import Path


def _safe_simname(s: str, max_len: int = 20) -> str:
    s = str(s).strip()
    if not s:
        raise ValueError("Empty simname")
    if len(s) <= max_len:
        return s
    return s[:max_len]


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
    return f"{indent}{str(new_value).rstrip()}     !{comment.rstrip()}\n"


def _patch_sim_template(
    template_sim: Path,
    out_sim: Path,
    simname: str,
    n0: int,
    filexy: str,
    tts: float,
    tt0: float,
    filetimeseries: str,
):
    lines = template_sim.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)

    def _edit_by_marker(marker: str, value: str):
        for i, ln in enumerate(lines):
            if marker in ln:
                lines[i] = _replace_value_keep_comment(ln, value)
                return True
        return False

    # simname: first value line after INTRO header
    simname_set = False
    for i, ln in enumerate(lines):
        if i == 0:
            continue
        if ln.lstrip().startswith("-"):
            continue
        if "simulation name" in ln and "!" in ln:
            lines[i] = _replace_value_keep_comment(ln, simname)
            simname_set = True
            break
    if not simname_set:
        raise ValueError(f"Cannot find simulation name line in template: {template_sim}")

    if not _edit_by_marker("! N0", str(int(n0))):
        raise ValueError(f"Cannot find N0 line in template: {template_sim}")
    if not _edit_by_marker("! filexy", str(filexy)):
        raise ValueError(f"Cannot find filexy line in template: {template_sim}")
    if not _edit_by_marker("! TTs", f"{float(tts)}"):
        raise ValueError(f"Cannot find TTs line in template: {template_sim}")
    if not _edit_by_marker("! tt0", f"{float(tt0)}"):
        raise ValueError(f"Cannot find tt0 line in template: {template_sim}")
    filetimeseries_quoted = f"'{str(filetimeseries)}'"
    if not _edit_by_marker("! filetimeseries", filetimeseries_quoted):
        raise ValueError(f"Cannot find filetimeseries line in template: {template_sim}")

    out_sim.write_text("".join(lines), encoding="utf-8")


def _run_py(script: Path, args: list[str], cwd: Path):
    cmd = [sys.executable, str(script)] + list(args)
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(description="One-click MCMM forecast rollout (timeseries -> sim -> run -> parse)")
    parser.add_argument("--start", default="202501")
    parser.add_argument("--end", default="203012")
    parser.add_argument("--initial_month", default="202412")

    parser.add_argument("--scenario", default="S0")
    parser.add_argument("--member", type=int, default=0)

    parser.add_argument(
        "--wd_method",
        default="climatology",
        choices=["climatology", "q_scaled", "ahg", "ahg_monthly", "delayed_ssm"],
        help="How to generate W/D for MCMM timeseries (default: climatology). Use q_scaled/ahg to propagate scenario Q perturbations.",
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
        help="Optional W/D clip diagnostics csv. Use 'auto' to write into run input/future_timeseries (default: none).",
    )

    parser.add_argument(
        "--e_rate_method",
        default="median",
        choices=["median", "monthly_median", "q_scaled"],
        help="How to generate E_rate for MCMM timeseries (default: median).",
    )
    parser.add_argument(
        "--e_rate_base",
        default="monthly_median",
        choices=["median", "monthly_median"],
        help="Base E_rate profile before q scaling (only used when --e_rate_method q_scaled).",
    )
    parser.add_argument(
        "--e_rate_q_power",
        type=float,
        default=1.0,
        help="E_rate scaling exponent for Q ratio (only used when --e_rate_method q_scaled).",
    )
    parser.add_argument(
        "--e_rate_q_clip_min",
        type=float,
        default=0.5,
        help="Min clip for Q ratio when scaling E_rate (only used when --e_rate_method q_scaled).",
    )
    parser.add_argument(
        "--e_rate_q_clip_max",
        type=float,
        default=2.0,
        help="Max clip for Q ratio when scaling E_rate (only used when --e_rate_method q_scaled).",
    )

    parser.add_argument("--n0", type=int, default=230)
    parser.add_argument("--tt0", type=float, default=0.0)
    parser.add_argument("--tts", type=float, default=6.0, help="Simulation duration in years (dimensionless)")

    parser.add_argument("--template_sim", default="my_jingjiang.sim")
    parser.add_argument("--simname", default=None, help="Override simulation name (will be truncated to <=20 chars)")

    parser.add_argument(
        "--runs_dir",
        default=None,
        help="Directory to store isolated run workdirs (default: <mcmm_root>/runs)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional tag appended to run dir name (useful to avoid collisions)",
    )

    args = parser.parse_args()

    mcmm_root = Path(__file__).resolve().parent
    input_dir = mcmm_root

    runs_dir = Path(args.runs_dir) if args.runs_dir else (mcmm_root / "runs")
    if not runs_dir.is_absolute():
        runs_dir = (Path.cwd() / runs_dir).resolve()

    base_simname = (
        str(args.simname)
        if args.simname
        else f"f{str(args.start)[:4]}_{str(args.end)[:4]}_{str(args.scenario)}_m{int(args.member)}"
    )
    simname = _safe_simname(base_simname, max_len=20)

    run_dir_name = simname
    if args.tag:
        run_dir_name = f"{run_dir_name}_{str(args.tag).strip()}"

    workdir = runs_dir / run_dir_name
    if workdir.exists():
        raise FileExistsError(
            f"Run dir already exists: {workdir}. Use --tag to create a new run dir, or delete it manually."
        )

    work_input = workdir / "input"
    work_output = workdir / "output"
    work_temp = workdir / "temp"
    work_input.mkdir(parents=True, exist_ok=True)
    work_output.mkdir(parents=True, exist_ok=True)
    work_temp.mkdir(parents=True, exist_ok=True)

    # 1) Build future timeseries into isolated input/
    ts_dir = work_input / "future_timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)
    ts_name = f"jingjiang_monthly_{args.start}_{args.end}_{args.scenario}_m{int(args.member)}.csv"
    ts_out = ts_dir / ts_name

    wd_clip_diag_csv = None
    if args.wd_clip_diag_csv is not None:
        s = str(args.wd_clip_diag_csv).strip()
        if s.lower() == "auto":
            wd_clip_diag_csv = ts_dir / f"wd_clip_diag_{str(args.scenario)}_m{int(args.member)}.csv"
        elif s:
            wd_clip_diag_csv = Path(s)

    ts_args = [
        "--start",
        str(args.start),
        "--end",
        str(args.end),
        "--scenario",
        str(args.scenario),
        "--member",
        str(int(args.member)),
        "--wd_method",
        str(args.wd_method),
        "--wd_ssm_sigma_scale",
        str(float(args.wd_ssm_sigma_scale)),
        "--wd_ssm_sample",
        str(int(args.wd_ssm_sample)),
        "--e_rate_method",
        str(args.e_rate_method),
        "--e_rate_base",
        str(args.e_rate_base),
        "--e_rate_q_power",
        str(float(args.e_rate_q_power)),
        "--e_rate_q_clip_min",
        str(float(args.e_rate_q_clip_min)),
        "--e_rate_q_clip_max",
        str(float(args.e_rate_q_clip_max)),
        "--wd_noise_ln_sigma",
        str(float(args.wd_noise_ln_sigma)),
        "--e_rate_noise_ln_sigma",
        str(float(args.e_rate_noise_ln_sigma)),
        "--wd_clip_ref",
        str(args.wd_clip_ref),
        "--wd_clip_q_low",
        str(float(args.wd_clip_q_low)),
        "--wd_clip_q_high",
        str(float(args.wd_clip_q_high)),
        "--wd_clip_pad_frac",
        str(float(args.wd_clip_pad_frac)),
        "--out",
        str(ts_out),
    ]
    if wd_clip_diag_csv is not None:
        ts_args += ["--wd_clip_diag_csv", str(wd_clip_diag_csv)]
    if args.rng_seed is not None:
        ts_args += ["--rng_seed", str(int(args.rng_seed))]
    if args.wd_ssm_params is not None:
        ts_args += ["--wd_ssm_params", str(args.wd_ssm_params)]

    _run_py(
        script=input_dir / "build_mcmm_timeseries_from_forcing.py",
        args=ts_args,
        cwd=mcmm_root,
    )

    # 2) Build initial centerline xy into isolated input/
    xy_out = work_input / "jingjiang_centerline.xy"
    _run_py(
        script=input_dir / "convert_centerline.py",
        args=[
            "--initial_month",
            str(args.initial_month),
            "--n_out",
            str(int(args.n0)),
            "--output",
            str(xy_out),
        ],
        cwd=mcmm_root,
    )

    # 3) Patch template sim and write into isolated input/
    template_sim = input_dir / Path(str(args.template_sim)).name
    if not template_sim.exists():
        raise FileNotFoundError(f"Template sim not found: {template_sim}")

    sim_filename = f"{simname}.sim"
    sim_out = work_input / sim_filename

    filetimeseries_rel = f"future_timeseries/{ts_name}"
    _patch_sim_template(
        template_sim=template_sim,
        out_sim=sim_out,
        simname=simname,
        n0=int(args.n0),
        filexy="jingjiang_centerline.xy",
        tts=float(args.tts),
        tt0=float(args.tt0),
        filetimeseries=filetimeseries_rel,
    )

    # 4) Run MCMM
    log_path = work_output / f"{simname}_mcmm.log"
    _run_py(
        script=input_dir / "run_mcmm_sim.py",
        args=[
            "--sim",
            sim_filename,
            "--workdir",
            str(workdir),
            "--log",
            str(log_path),
        ],
        cwd=mcmm_root,
    )

    # 5) Parse output into UTM 1000pts
    _run_py(
        script=input_dir / "parse_mcmm_output.py",
        args=[
            "--start_yyyymm",
            str(args.initial_month),
            "--end_yyyymm",
            str(args.end),
            "--initial_month",
            str(args.initial_month),
            "--simname",
            simname,
            "--output_dir",
            str(work_output),
            "--qc_reference",
            "--orient_to_reference",
        ],
        cwd=mcmm_root,
    )

    print("\nDONE")
    print(f"  simname          = {simname}")
    print(f"  run_dir          = {workdir}")
    print(f"  timeseries_csv   = {ts_out}")
    print(f"  init_centerline  = {xy_out}")
    print(f"  mcmm_log         = {log_path}")
    print(f"  parsed_dir       = {work_output / 'mcmm_centerlines_utm_1000pts'}")


if __name__ == "__main__":
    main()
