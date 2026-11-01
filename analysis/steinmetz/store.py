"""Atomic result files: a stage's output exists only once it is complete."""
import json
import os
import tempfile
import zlib
from pathlib import Path

import numpy as np


def save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def load_json(path):
    return json.loads(Path(path).read_text())


def session_rng(seed, name, salt=0):
    """A generator that depends only on the session and purpose, so resumed runs give the same draws."""
    return np.random.default_rng([seed, zlib.crc32(name.encode()), salt])
