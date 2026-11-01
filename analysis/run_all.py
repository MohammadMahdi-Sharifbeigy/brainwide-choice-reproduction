"""Run the whole analysis, then make all figures and the report.

    env\\Scripts\\python.exe analysis\\run_all.py                     full dataset
    env\\Scripts\\python.exe analysis\\run_all.py --session NAME      one session (smoke run)
    env\\Scripts\\python.exe analysis\\run_all.py --figures-only      rebuild figures from saved results
"""
import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")  # one BLAS thread per worker, to avoid oversubscribing cores

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figures.build import make_all  # noqa: E402
from steinmetz import pipeline, store  # noqa: E402
from steinmetz.config import ANALYSIS, Config  # noqa: E402
from steinmetz.data import list_sessions  # noqa: E402

DEFAULT_PEAK_GB = 3.0  # used until a smoke run has measured the real peak per worker


def expected_peak(cfg):
    """Peak per worker for the largest session, scaled from the smoke run when there is one."""
    smoke = ANALYSIS / "results_smoke"
    if not (smoke / "timing.json").exists():
        return DEFAULT_PEAK_GB
    peak = max(v["peak_gb"] for v in store.load_json(smoke / "timing.json")["stages"].values())
    size = lambda s: (cfg.data / s / "spikes.times.npy").stat().st_size
    done = [p.stem for p in (smoke / "base").glob("*.npz")]
    return pipeline.scaled_peak(peak, max(size(s) for s in done), max(size(s) for s in list_sessions(cfg.data)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", help="run one session only, for a smoke run")
    ap.add_argument("--workers", type=int, help="override the automatic worker count")
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    sessions = None
    if args.session:
        cfg = replace(cfg, results=ANALYSIS / "results_smoke", output=ANALYSIS / "output_smoke", min_subjects=1)
        sessions = [args.session]
    if not args.figures_only:
        workers = args.workers or pipeline.auto_workers(expected_peak(cfg))
        print(f"workers: {workers}", flush=True)
        pipeline.run(cfg, sessions, workers)
    failed = make_all(cfg)
    if failed:
        print(f"figures that failed: {sorted(failed)}")
        sys.exit(1)
    print(f"done: {cfg.output}")


if __name__ == "__main__":
    main()
