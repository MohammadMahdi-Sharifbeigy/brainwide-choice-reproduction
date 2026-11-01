"""Search all neurons for the ones closest to the paper's examples. Reads saved results and changes no analysis.

Run this once, then rebuild the figures with run_all.py. Edit results/example_neurons.json by hand to swap a neuron.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from steinmetz import examples  # noqa: E402
from steinmetz.config import Config  # noqa: E402
from steinmetz.results import Results  # noqa: E402


def main():
    cfg = Config()
    chosen = examples.select(Results(cfg))
    for slot, e in chosen.items():
        print(f"{slot:28s} {e['session']:24s} {e['cluster']}")
    print(f"scores: {cfg.results / 'example_scores.csv'}")


if __name__ == "__main__":
    main()
