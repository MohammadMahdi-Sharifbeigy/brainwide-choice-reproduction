"""Allen hierarchy, the paper's 42 regions and neuron sets."""
import json
from functools import lru_cache

import numpy as np

from .config import ANATOMY_ORDER, ATLAS_DIR, MIN_NEURONS_REGION


@lru_cache(maxsize=1)
def structure_tree():
    """acronym -> (id, parent acronym), and id -> acronym."""
    root = json.loads((ATLAS_DIR / "structure_graph.json").read_text())["msg"][0]
    by_acr, by_id = {}, {}
    stack = [(root, None)]
    while stack:
        node, parent = stack.pop()
        by_acr[node["acronym"]] = (node["id"], parent)
        by_id[node["id"]] = node["acronym"]
        stack.extend((c, node["acronym"]) for c in node["children"])
    return by_acr, by_id


def to_paper_region(label):
    """Nearest ancestor of an Allen label among the 42 regions, or '' when not localizable."""
    by_acr, _ = structure_tree()
    while label is not None:
        if label in ANATOMY_ORDER:
            return label
        label = by_acr[label][1] if label in by_acr else None
    return ""


def neuron_regions(session):
    """Raw label and paper region for every cluster, from the peak channel."""
    labels = session.channel_labels()
    raw = labels[session.peak_channel()]
    table = {u: to_paper_region(u) for u in np.unique(labels)}
    return raw, np.array([table[r] for r in raw])


def region_counts(regions, mask):
    names, n = np.unique(regions[mask], return_counts=True)
    return dict(zip(names.tolist(), n.tolist()))


def kept_regions(per_session, subjects, min_subjects):
    """Regions meeting the >=13 per session and >=min_subjects rule.

    per_session: session -> {region: base neuron count}. Returns session -> set of kept regions.
    """
    seen = {}
    for s, counts in per_session.items():
        for r, n in counts.items():
            if r and n >= MIN_NEURONS_REGION:
                seen.setdefault(r, set()).add(subjects[s])
    good = {r for r, mice in seen.items() if len(mice) >= min_subjects}
    return {s: {r for r, n in c.items() if r in good and n >= MIN_NEURONS_REGION} for s, c in per_session.items()}


def atlas_labels(session, volume, id_table):
    """Paper region per channel when read from the atlas at the channel's CCF position (the bounded label check)."""
    xyz = np.round(session.channel_ccf() / 25).astype(int)
    xyz = np.clip(xyz, 0, np.array(volume.shape) - 1)
    ids = volume[xyz[:, 0], xyz[:, 1], xyz[:, 2]]
    return np.array([to_paper_region(id_table.get(int(i), "")) for i in ids])
