"""Streaming cloud-in-cell (CIC) mesh painter — pure numpy, memory-safe.

Deposits particle positions onto a periodic ``nmesh^3`` grid. Designed to be fed
in **chunks** (accumulate into a preallocated ``out`` array) so a huge particle
load never has to be resident in memory at once — the peak footprint is the mesh
plus one chunk. This is the lightweight default backend for the IC loader (an
optional ``nbodykit`` backend can be added later); it needs no MPI and runs on a
NERSC login node.

CIC weighting: each particle contributes to the 8 nearest cells with weights
``(1-d)`` / ``d`` per axis (``d`` = fractional offset from the lower cell),
wrapped periodically. Total deposited mass equals the particle count.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def cic_paint(
    positions,
    nmesh: int,
    boxsize: Optional[float] = None,
    out: Optional[np.ndarray] = None,
    weights=None,
) -> np.ndarray:
    """Deposit ``positions`` onto an ``nmesh^3`` mesh by CIC; return the mesh.

    Parameters
    ----------
    positions : array (N, 3)
        Particle positions. In mesh units ``[0, nmesh)`` if ``boxsize`` is None,
        otherwise physical coordinates in ``[0, boxsize)`` (scaled by
        ``nmesh / boxsize``).
    nmesh : int
        Cells per side.
    boxsize : float, optional
        If given, ``positions`` are physical and rescaled to mesh units.
    out : ndarray (nmesh, nmesh, nmesh), optional
        Accumulate into this float64 array (for chunked/streaming use). If None,
        a fresh zero array is allocated.
    weights : array (N,), optional
        Per-particle weights (default 1).

    Returns
    -------
    ndarray
        The (accumulated) mass grid, float64.
    """
    if out is None:
        out = np.zeros((nmesh, nmesh, nmesh), dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3); got {pos.shape}")
    if boxsize is not None:
        pos = pos * (nmesh / float(boxsize))

    i = np.floor(pos).astype(np.int64)        # lower cell index per axis, (N,3)
    d = pos - i                                # fractional offset in [0,1), (N,3)
    w = np.ones(pos.shape[0]) if weights is None else np.asarray(weights, float)

    flat = out.reshape(-1)
    n2 = nmesh * nmesh
    for ox in (0, 1):
        wx = (1.0 - d[:, 0]) if ox == 0 else d[:, 0]
        ix = (i[:, 0] + ox) % nmesh
        for oy in (0, 1):
            wy = (1.0 - d[:, 1]) if oy == 0 else d[:, 1]
            iy = (i[:, 1] + oy) % nmesh
            for oz in (0, 1):
                wz = (1.0 - d[:, 2]) if oz == 0 else d[:, 2]
                iz = (i[:, 2] + oz) % nmesh
                idx = ix * n2 + iy * nmesh + iz
                flat += np.bincount(idx, weights=w * wx * wy * wz, minlength=flat.size)
    return out


def to_overdensity(rho: np.ndarray) -> np.ndarray:
    """Overdensity ``delta = rho/<rho> - 1`` (mean over all cells)."""
    mean = rho.mean()
    if mean == 0:
        raise ValueError("cannot form overdensity: mesh is empty (mean density 0)")
    return rho / mean - 1.0
