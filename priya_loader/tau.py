"""Load Lyman-alpha optical-depth (``tau``) fields from the gridded fake_spectra
HDF5 product (``lya_forest_spectra_grid_480.hdf5``).

The product stores HI Lyman-alpha optical depth at ``tau/H/1/1215`` with shape
``(Nlos, nbins)``, where ``Nlos = 3 * ngrid**2`` sightlines are laid out as a
regular ``ngrid x ngrid`` transverse grid along **three** line-of-sight axes
(x, y, z), stored contiguously in that order. ``nbins`` is the number of
velocity pixels along the LOS and is **redshift-dependent**.

:func:`load_tau_grid` returns the RAW tau for one chosen axis, reshaped (C-order,
verified against the file's ``cofm``) to a ``(ngrid, ngrid, nbins)`` cube — never
resampled or rebinned. Flux ``F = exp(-tau)`` and ``delta_flux = F/<F> - 1`` are
optional derived helpers, since Roger's pipeline consumes ``tau`` directly.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Union

import h5py
import numpy as np

from . import units

StrPath = Union[str, os.PathLike]

TAU_KEY = "tau/H/1/1215"
N_AXES = 3   # x, y, z line-of-sight directions


@dataclass
class TauGrid:
    """Raw Lyman-alpha optical depth on one LOS axis of a snapshot."""

    tau: np.ndarray        # (ngrid, ngrid, nbins) float32 — RAW optical depth
    redshift: float
    axis: int              # 1, 2, 3  <-> x, y, z
    ngrid: int             # transverse grid per side (480 for the PRIYA product)
    nbins: int             # LOS velocity pixels (redshift-dependent)
    box: float             # Mpc/h
    meta: dict = field(default_factory=dict, repr=False)


def load_tau_grid(path: StrPath, axis: int = 1) -> TauGrid:
    """Load the raw tau cube for one line-of-sight ``axis`` (1=x, 2=y, 3=z)."""
    if axis not in (1, 2, 3):
        raise ValueError(f"axis must be 1, 2 or 3 (x/y/z); got {axis!r}")
    with h5py.File(path, "r") as f:
        hdr = f["Header"].attrs
        redshift = float(hdr["redshift"])
        nbins = int(hdr["nbins"])
        box_kpc_h = float(hdr["box"])
        keys = set(hdr.keys())
        hubble = float(hdr["hubble"]) if "hubble" in keys else None
        hz = float(hdr["Hz"]) if "Hz" in keys else None
        omegam = float(hdr["omegam"]) if "omegam" in keys else None
        omegab = float(hdr["omegab"]) if "omegab" in keys else None
        omegal = float(hdr["omegal"]) if "omegal" in keys else None
        dset = f[TAU_KEY]
        nlos = dset.shape[0]
        ngrid = int(round(math.sqrt(nlos / N_AXES)))
        if ngrid * ngrid * N_AXES != nlos:
            raise ValueError(
                f"{path}: tau has {nlos} sightlines, not 3*ngrid^2 for any integer ngrid"
            )
        per_axis = ngrid * ngrid
        sl = slice((axis - 1) * per_axis, axis * per_axis)
        block = dset[sl]                                  # (per_axis, nbins), raw
    tau = block.reshape(ngrid, ngrid, nbins)             # C-order (verified vs cofm)

    dv_kms = None
    if hz is not None and hubble is not None:
        dv_kms = hz * (box_kpc_h / 1000.0 / hubble) / (1.0 + redshift) / nbins
    meta = {
        "path": str(path),
        "ngrid": ngrid,
        "nbins": nbins,
        "hubble": hubble,
        "omegam": omegam,
        "omegab": omegab,
        "omegal": omegal,
        "Hz": hz,
        "transverse_kpc_h": box_kpc_h / ngrid,
        "dv_kms": dv_kms,
    }
    return TauGrid(
        tau=tau,
        redshift=redshift,
        axis=axis,
        ngrid=ngrid,
        nbins=nbins,
        box=units.kpc_h_to_mpc_h(box_kpc_h),
        meta=meta,
    )


# --- optional derived helpers -------------------------------------------------
def to_flux(tau: np.ndarray) -> np.ndarray:
    """Transmitted flux ``F = exp(-tau)``."""
    return np.exp(-tau)


def mean_flux(tau: np.ndarray) -> float:
    """Mean transmitted flux ``<F> = <exp(-tau)>`` over the array."""
    return float(np.exp(-tau).mean())


def to_delta_flux(tau: np.ndarray, mean_flux: float = None) -> np.ndarray:
    """Flux contrast ``delta_F = F/<F> - 1``. If ``mean_flux`` is None, use the
    array's own mean (per-array normalisation)."""
    flux = np.exp(-tau)
    mf = flux.mean() if mean_flux is None else mean_flux
    return flux / mf - 1.0
