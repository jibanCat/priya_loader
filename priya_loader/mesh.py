"""Streaming cloud-in-cell (CIC) mesh painter — pure numpy, memory-safe.

**Not nbodykit / pmesh.** This is a self-contained implementation of the
standard cloud-in-cell mass-assignment scheme (Hockney & Eastwood 1981, "Computer
Simulation Using Particles", §5-3): a `np.bincount` scatter of the 8 tri-linear
corner weights with periodic wrap. It has **no third-party meshing dependency**
(no nbodykit, pmesh, or MPI) — only numpy — so there is no external commit to
pin. It matches nbodykit's ``resampler="cic"`` *uncompensated* result up to the
CIC window (we ship raw; see below). ``backend="nbodykit"`` in the IC loader is
not implemented.

Deposits particle positions onto a periodic ``nmesh^3`` grid. Designed to be fed
in **chunks** (accumulate into a preallocated ``out`` array) so a huge particle
load never has to be resident at once. Peak memory is ``~2 * nmesh^3`` float64
(the persistent grid + one transient ``bincount`` array; the 8 corner bincounts
are sequential, not simultaneous) plus one chunk's working set (``~120 B *
chunk_size``). Recommended ``nmesh <= 512`` on a NERSC login node
(512^3 ~ 2 GiB, 1024^3 ~ 16 GiB). It needs no MPI.

The result is a **raw, uncompensated** CIC field: it carries the CIC window
``W(k) = prod_i sinc^2(k_i * dx / 2)`` with ``dx = box/nmesh`` (Jing 2005,
arXiv:astro-ph/0409240, eqs 3,7; Hockney & Eastwood 1981 §5-3). For a finite-k
cross-spectrum (e.g. IC x flux for the bias), deconvolve this ``sinc^2`` before
use; at ``k -> 0`` the window -> 1. (Shipping raw is the deliberate design; an
optional ``nbodykit`` / compensated backend can be added later.)

Algorithm (per particle at mesh-unit position ``p = position/boxsize * nmesh``):
  1. lower cell ``i = floor(p)``; fractional offset ``d = p - i`` in ``[0,1)``;
  2. for each of the 8 corners ``o in {0,1}^3``, weight ``w = prod_axis (1-d if
     o=0 else d)`` is added to cell ``(i + o) mod nmesh`` (periodic wrap).
The 8 weights sum to 1, so total deposited mass = particle count (mass-conserving).
The grid node sits at integer cell coordinate, i.e. position ``j*box/nmesh`` —
matching fake_spectra's ``cofm`` grid so the IC mesh co-registers with tau.

Implementation: the only loop is over the 8 fixed corners; for each, all
particles are scattered in one ``np.bincount`` (O(N), vectorized). This is the
same kernel as nbodykit/pmesh ``resampler="cic"`` (uncompensated, non-interlaced)
— verified bit-for-bit against an explicit reference CIC in the tests, and
against nbodykit itself where it is installed.
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
    if nmesh < 1:
        raise ValueError(f"nmesh must be >= 1; got {nmesh}")
    if out is None:
        out = np.zeros((nmesh, nmesh, nmesh), dtype=np.float64)
    else:
        if out.shape != (nmesh, nmesh, nmesh):
            raise ValueError(f"out shape {out.shape} != ({nmesh},)*3")
        if out.dtype != np.float64:
            raise TypeError(f"out must be float64 (got {out.dtype}) to avoid precision loss")
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
