"""Power benchmark: log the USB meter on this PC while the Pi runs the sustained benchmark.

Run from the Version1 folder on the PC, venv active, meter connected (see power_log.py):
    python bench/power_run.py                              # all 9 configs, 60 s each
    python bench/power_run.py --seconds 20 --cells rec-fp32_live-fp32   # quick check

What happens:
  1. power_log.py starts logging the meter to power.csv (PC clock).
  2. Over SSH, the Pi runs `bench.py --power`: an idle baseline, then per config
     load -> warm-up -> sustained run -> idle cool-down. Each phase boundary arrives as a
     PHASE line carrying the Pi's timestamp; this script adds the PC's receive time.
  3. The logger is stopped and eval/power_analysis.py computes power per config.

Writes results/raw/power/<label>/<run_id>/:
    power.csv     meter samples, 100/s (t_unix, V, A, W)
    phases.csv    phase boundaries with t_pi and t_pc
    cells.jsonl   one RESULT per config (frames, latency, temperature, throttling, RSS)
    run.json      settings, code hashes on both machines, logger status
    pi_stdout.log everything the Pi printed
The Pi needs key-based SSH (no password prompt) and the same bench.py/pipeline.py as the PC.
"""
import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

version1_folder = Path(__file__).resolve().parents[1]
power_raw = version1_folder / "results" / "raw" / "power"
CODE_FILES = ["bench/bench.py", "app/pipeline.py"]


def sha256(path):
    """Hash with LF line endings, matching the Pi's checkout of the same file."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def ssh(host, command, **kwargs):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, command], **kwargs)


def wait_for_samples(csv_path, logger, minimum=100, timeout=60):
    """Block until the logger has written `minimum` samples, so the baseline is covered."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if logger.poll() is not None:
            sys.exit("power_log.py exited before logging anything (is the meter connected?)")
        if csv_path.exists():
            with open(csv_path) as f:
                if sum(1 for _ in f) > minimum:
                    return
        time.sleep(0.5)
    logger.terminate()
    sys.exit(f"No meter samples after {timeout} s; check the micro-USB data cable.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="simion@simiPi")
    ap.add_argument("--remote-dir", default="~/facial-recognition-optimization/Version1")
    ap.add_argument("--label", default="pi")
    ap.add_argument("--note", default="", help="free text saved in run.json (e.g. 'case open, no fan')")
    # passed through to bench.py --power
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--idle", type=float, default=60)
    ap.add_argument("--gap", type=float, default=30)
    ap.add_argument("--cool-to", type=float, default=50)
    ap.add_argument("--max-cool", type=float, default=300)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--cells", default="")
    ap.add_argument("--reverse", action="store_true", help="run the configs in reverse order")
    args = ap.parse_args()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out = power_raw / args.label / run_id
    out.mkdir(parents=True, exist_ok=True)
    stop_file = out / "stop"

    # Same code on both machines, or the run measures something else
    remote_hashes = ssh(args.host, f"cd {args.remote_dir} && sha256sum {' '.join(CODE_FILES)}",
                        capture_output=True, text=True)
    if remote_hashes.returncode != 0:
        sys.exit(f"SSH to {args.host} failed: {remote_hashes.stderr.strip()}")
    pi_hashes = {line.split()[1]: line.split()[0] for line in remote_hashes.stdout.splitlines()}
    pc_hashes = {f: sha256(version1_folder / f) for f in CODE_FILES}
    mismatched = [f for f in CODE_FILES if pi_hashes.get(f) != pc_hashes[f]]
    if mismatched:
        sys.exit(f"Pi code differs from the PC for {mismatched}; copy or pull it first.")

    logger = subprocess.Popen([sys.executable, str(version1_folder / "bench" / "power_log.py"),
                               "--out", str(out / "power.csv"), "--stop-file", str(stop_file)])
    wait_for_samples(out / "power.csv", logger)
    print(f"Meter logging -> {out / 'power.csv'}")

    bench_args = ["--label", args.label, "--power", "--seconds", args.seconds, "--warmup", args.warmup,
                  "--idle", args.idle, "--gap", args.gap, "--cool-to", args.cool_to,
                  "--max-cool", args.max_cool, "--repeats", args.repeats]
    if args.cells:
        bench_args += ["--cells", args.cells]
    if args.reverse:
        bench_args.append("--reverse")
    remote = (f"cd {args.remote_dir} && PYTHONUNBUFFERED=1 .venv/bin/python bench/bench.py "
              + " ".join(shlex.quote(str(a)) for a in bench_args))

    started = time.time()
    pi_status = None
    with open(out / "phases.csv", "w", newline="") as phases, \
            open(out / "cells.jsonl", "w") as cells, \
            open(out / "pi_stdout.log", "w") as log:
        phase_writer = csv.writer(phases)
        phase_writer.writerow(["cell", "phase", "event", "t_pi", "t_pc", "extra"])
        pi = subprocess.Popen(["ssh", "-o", "BatchMode=yes", args.host, remote],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in pi.stdout:
                t_pc = time.time()
                log.write(line)
                log.flush()
                if line.startswith("PHASE "):
                    m = json.loads(line[6:])
                    extra = {k: v for k, v in m.items() if k not in ("cell", "phase", "event", "t")}
                    phase_writer.writerow([m["cell"], m["phase"], m["event"], f"{m['t']:.3f}",
                                           f"{t_pc:.3f}", json.dumps(extra)])
                    phases.flush()
                    print(f"[{t_pc - started:7.1f} s] {m['cell']:28s} {m['phase']:9s} {m['event']}"
                          + (f"  {extra}" if extra else ""))
                elif line.startswith("RESULT "):
                    cells.write(line[7:])
                    cells.flush()
                else:
                    print("  pi> " + line.rstrip())
            pi_status = pi.wait()
        except KeyboardInterrupt:
            print("\nInterrupted: stopping the Pi benchmark and the logger.")
            pi.terminate()
            ssh(args.host, "pkill -f 'bench/bench.py --label'", capture_output=True)
            pi_status = "interrupted"
        finally:
            # Keep logging a few seconds past the last idle marker, then stop cleanly
            time.sleep(3)
            stop_file.touch()
            try:
                logger_status = logger.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.terminate()
                logger_status = "killed"
            stop_file.unlink(missing_ok=True)

    run_info = {
        "run_id": run_id, "label": args.label, "host": args.host, "note": args.note,
        "settings": {k: v for k, v in vars(args).items() if k not in ("host", "remote_dir", "note")},
        "code_sha256": pc_hashes, "pi_exit": pi_status,
        "logger_exit": logger_status,
        "logger_ok": logger_status == 0,
        "meter": "FNIRSI FNB58, inline on the Pi's USB-C supply; 100 samples/s",
    }
    (out / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")
    if pi_status != 0 or logger_status != 0:
        sys.exit(f"Run incomplete (Pi exit {pi_status}, logger exit {logger_status}); "
                 f"raw files kept in {out}")

    subprocess.run([sys.executable, str(version1_folder / "eval" / "power_analysis.py"), str(out)], check=True)


if __name__ == "__main__":
    main()
