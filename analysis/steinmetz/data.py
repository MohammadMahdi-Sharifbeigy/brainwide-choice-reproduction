"""Read one session's files."""
import csv
from pathlib import Path

import numpy as np


def list_sessions(data_dir):
    return sorted(p.name for p in Path(data_dir).iterdir() if p.is_dir())


def subject_of(session):
    return session.split("_")[0]


class Session:
    """Lazy access to a session folder; `.a(name)` returns a raveled array."""

    def __init__(self, data_dir, name):
        self.path = Path(data_dir) / name
        self.name = name
        self.subject = subject_of(name)

    def raw(self, name):
        return np.load(self.path / f"{name}.npy", allow_pickle=False)

    def a(self, name):
        return self.raw(name).ravel()

    def trials(self):
        return {
            "stim": self.a("trials.visualStim_times"),
            "cue": self.a("trials.goCue_times"),
            "response": self.a("trials.response_times"),
            "fb_time": self.a("trials.feedback_times"),
            "fb": self.a("trials.feedbackType"),
            "choice": self.a("trials.response_choice"),
            "cl": self.a("trials.visualStim_contrastLeft"),
            "cr": self.a("trials.visualStim_contrastRight"),
            "included": self.a("trials.included").astype(bool),
            "intervals": self.raw("trials.intervals"),
        }

    def passive(self):
        return {
            "times": self.a("passiveVisual.times"),
            "cl": self.a("passiveVisual.contrastLeft"),
            "cr": self.a("passiveVisual.contrastRight"),
        }

    def moves(self):
        iv = self.raw("wheelMoves.intervals")
        return iv[:, 0], iv[:, 1], self.a("wheelMoves.type").astype(int)

    def spikes(self):
        return self.a("spikes.times"), self.a("spikes.clusters").astype(np.int64)

    def channel_table(self):
        with open(self.path / "channels.brainLocation.tsv", newline="") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    def channel_labels(self):
        return np.array([r["allen_ontology"].strip() for r in self.channel_table()])

    def channel_ccf(self):
        return np.array([[float(r[k]) for k in ("ccf_ap", "ccf_dv", "ccf_lr")] for r in self.channel_table()])

    def peak_channel(self):
        """Zero-based index into the channel table; the file is one-based."""
        return self.a("clusters.peakChannel").astype(int) - 1

    def wheel(self):
        """Wheel position in mm and its time base."""
        pos = self.a("wheel.position")
        st = self.raw("wheel.timestamps")
        t = np.interp(np.arange(len(pos)), st[:, 0], st[:, 1])
        return t, pos

    def series(self, name):
        """A sampled signal such as eye.area with its per-sample times."""
        v = self.a(f"{name[0]}.{name[1]}")
        st = self.raw(f"{name[0]}.timestamps")
        return np.interp(np.arange(len(v)), st[:, 0], st[:, 1]), v
