import numpy as np
import torch


SG_THRESH = 1000    # sub-graph threshold = 1000 atoms


def graph_cut(x, radius_min=10.0, radius_max=20.0, xc=None, block_id=None):
    """
    :param x: coordinates of all atoms, (N, 3), unit: Angstrom
    :param radius_min: subgraph for computing grads within a sphere of radius_min
    :param radius_max: subgraph for constructing edges within a sphere of radius_max
    :param block_id: optional original atom-to-block IDs. If provided, the
        outer crop is expanded to contain complete blocks.
    :return: max_indices, mask
    """
    N = x.shape[0]  # atoms
    if xc is None:
        center = np.random.randint(N)
        xc = x[center]

    dist = np.linalg.norm(x - xc, axis=-1)

    min_indices = np.where(dist < radius_min)[0]
    max_indices = np.where(dist < radius_max)[0]

    if block_id is not None:
        block_id = np.asarray(block_id)
        if block_id.shape[0] != N:
            raise ValueError("block_id must have one entry per atom")
        selected_blocks = np.unique(block_id[max_indices])
        max_indices = np.flatnonzero(np.isin(block_id, selected_blocks))

    mask = np.isin(max_indices, min_indices).astype(np.int64)

    return max_indices, mask
