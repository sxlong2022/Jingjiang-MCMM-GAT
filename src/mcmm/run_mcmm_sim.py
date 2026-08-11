import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run mcmm.exe with a .sim file name via stdin (non-interactive)")
    parser.add_argument("--sim", required=True, help="Sim file name (recommended: just the .sim name, e.g. my_jingjiang.sim)")
    parser.add_argument("--exe", default=None, help="Path to mcmm.exe (default: <mcmm_root>/mcmm.exe)")
    parser.add_argument("--workdir", default=None, help="Working directory (default: <mcmm_root>)")
    parser.add_argument("--log", default=None, help="Optional log file path")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    workdir = Path(args.workdir) if args.workdir else root
    if not workdir.is_absolute():
        workdir = (Path.cwd() / workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    (workdir / "input").mkdir(parents=True, exist_ok=True)
    (workdir / "output").mkdir(parents=True, exist_ok=True)
    (workdir / "temp").mkdir(parents=True, exist_ok=True)

    exe = Path(args.exe) if args.exe else (root / "mcmm.exe")
    if not exe.exists():
        raise FileNotFoundError(f"mcmm executable not found: {exe}")

    sim_name = Path(str(args.sim)).name
    stdin_payload = f"{sim_name}\n" + ("1\n" * 20)

    proc = subprocess.run(
        [str(exe)],
        cwd=str(workdir),
        input=stdin_payload,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    if args.log:
        log_path = Path(args.log)
        if not log_path.is_absolute():
            log_path = workdir / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(proc.stdout or "", encoding="utf-8", errors="ignore")

    if proc.stdout:
        sys.stdout.write(proc.stdout)

    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
