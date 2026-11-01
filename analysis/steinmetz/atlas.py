"""Allen atlas views for the brain maps."""
from functools import lru_cache

import numpy as np

from .config import ANATOMY_ORDER, ATLAS_DIR
from .regions import coarse_ids, structure_tree, to_paper_region

MIDLINE_UM = 5700
SAGITTAL_ML_MM = 1.7
RES_UM = 25


@lru_cache(maxsize=1)
def volume():
    import nrrd

    return nrrd.read(str(ATLAS_DIR / "annotation_25.nrrd"))[0]


def _to_region_index(ids):
    """Index into ANATOMY_ORDER per voxel, -1 for other tissue, -2 outside the brain."""
    _, by_id = structure_tree()
    u, inv = np.unique(ids, return_inverse=True)
    table = np.array([-2 if i == 0 else (ANATOMY_ORDER.index(to_paper_region(by_id.get(int(i), ""))) if to_paper_region(by_id.get(int(i), "")) else -1)
                      for i in u])
    return table[inv].reshape(ids.shape)


@lru_cache(maxsize=2)
def _ids(name):
    v = volume()
    if name == "top":
        first = (v != 0).argmax(1)
        return np.take_along_axis(v, first[:, None, :], axis=1)[:, 0, :]
    lr = int(round((MIDLINE_UM - SAGITTAL_ML_MM * 1000) / RES_UM))
    return v[:, :, lr].T


@lru_cache(maxsize=2)
def view(name):
    """Region index image: 'top' (AP x LR, surface structure) or 'sagittal' (DV x AP at 1.7 mm ML, left hemisphere)."""
    return _to_region_index(_ids(name))


@lru_cache(maxsize=2)
def tissue(name):
    """All structures at a coarse level, for the grey outlines of tissue outside the recorded regions."""
    ids = _ids(name)
    u, inv = np.unique(ids, return_inverse=True)
    table = coarse_ids()
    return np.array([table.get(int(i), int(i)) for i in u])[inv].reshape(ids.shape)


def colored(index_image, colors, other=(1.0, 1.0, 1.0, 1.0)):
    """RGBA image. colors maps a region name to RGBA; regions without a color and other tissue use `other`."""
    img = np.ones((*index_image.shape, 4))
    img[index_image == -1] = other
    img[index_image == -2] = (1, 1, 1, 0)
    for i, name in enumerate(ANATOMY_ORDER):
        m = index_image == i
        if m.any():
            img[m] = colors.get(name, (1, 1, 1, 1))
    return img


def outlines(index_image):
    """Boolean mask of region borders."""
    a = index_image
    edge = np.zeros(a.shape, bool)
    edge[:-1, :] |= a[:-1, :] != a[1:, :]
    edge[:, :-1] |= a[:, :-1] != a[:, 1:]
    return edge


def silhouette(axis):
    """Whole-brain outline mask for a projection along axis 0 (AP), 1 (DV) or 2 (LR)."""
    return (volume() != 0).any(axis)
