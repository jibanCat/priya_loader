"""Load Lyman-alpha optical-depth (``tau``) fields from the gridded fake_spectra
HDF5 product (``lya_forest_spectra_grid_480.hdf5``).

The product stores HI Lyman-alpha optical depth at ``tau/H/1/1215`` with shape
``(Nlos, nbins)``. The key encodes ``tau/<element>/<ion>/<lambda>`` =
H / ion 1 (HI) / 1215 Å (fake_spectra ``spectra.py:332-350`` @ v2.2.3 93d0e509).
This is the **3-axis grid product**: ``Nlos = 3 * ngrid**2`` sightlines form a
regular ``ngrid x ngrid`` transverse grid along each of the three line-of-sight
axes (x, y, z), the axis=1,2,3 blocks contiguous (``griddedspectra.py:42-60``).
``nbins`` (LOS velocity pixels) is **redshift-dependent**. The loader rejects a
single-axis file: ``ngrid = sqrt(Nlos/3)`` would not be integer, and the
``spectra/axis`` guard would fail.

:func:`load_tau_grid` returns the RAW tau for one chosen axis, reshaped (C-order)
to a ``(ngrid, ngrid, nbins)`` cube — never resampled or rebinned. A cheap
runtime check (``spectra/axis``) guards against a mis-ordered file.

Two things a downstream (bias) analysis must know, exposed on :class:`TauGrid`:

* **The tau is redshift-space** — ``fake_spectra`` bins absorption in velocity
  with ``vel = velfac*pos + pvel`` (Hubble flow + peculiar velocity;
  ``absorption.cpp:234`` @ v2.2.3 93d0e509), so the LOS (last) axis is a
  *velocity* axis and the field carries RSD (beta_F).
* **The two transverse cube indices are different physical coordinates per axis**
  (``cube_axes``): axis 1 -> ``(y, z, x_los)``, axis 2 -> ``(x, z, y_los)``,
  axis 3 -> ``(x, y, z_los)``, with the LOS coordinate last. Co-registering with
  an IC density mesh requires this mapping (a wrong transpose biases b_F).

Flux ``F = exp(-tau)`` and ``delta_flux = F/<F> - 1`` are optional derived
helpers (Roger's pipeline consumes ``tau`` directly).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import h5py
import numpy as np

from . import units

StrPath = Union[str, os.PathLike]

TAU_KEY = "tau/H/1/1215"
N_AXES = 3   # x, y, z line-of-sight directions

#: cube index -> physical coordinate, per LOS axis. (LOS coordinate is last.)
#:
#: Origin: fake_spectra ``griddedspectra.py:42-60`` @ v2.2.3 (commit 93d0e509),
#: which builds the transverse grid as
#:   axis==1: [0, nn, mm];  axis==2: [nn, 0, mm];  axis==3: [nn, mm, 0]
#: with ``for nn ... for mm ...`` (nn = outer/slow C-order index, mm = inner).
#: So cube[nn, mm] -> (y,z)/(x,z)/(x,y) with the lower-numbered transverse coord
#: as the slow index. Independently confirmed against ``spectra/cofm`` in a real
#: file (held LOS coord = 0; 250 ckpc/h spacing) for all three axes.
CUBE_AXES = {1: ("y", "z", "x"), 2: ("x", "z", "y"), 3: ("x", "y", "z")}


@dataclass
class TauGrid:
    """Raw, redshift-space Lyman-alpha optical depth on one LOS axis."""

    tau: np.ndarray              # (ngrid, ngrid, nbins) float32 — RAW optical depth
    redshift: float
    axis: int                    # 1, 2, 3  <-> x, y, z line of sight
    ngrid: int                   # transverse grid per side (480 for the product)
    nbins: int                   # LOS velocity pixels (redshift-dependent)
    box: float                   # Mpc/h
    cube_axes: Tuple[str, str, str]   # physical coord of each cube index; LOS last
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)


def load_tau_grid(path: StrPath, axis: int = 1) -> TauGrid:
    """Load the raw tau cube for one line-of-sight ``axis``.

    Parameters
    ----------
    path : str or os.PathLike
        A ``lya_forest_spectra_grid_480.hdf5`` file.
    axis : {1, 2, 3}
        Line of sight: 1=x, 2=y, 3=z. Selects one of the three 480x480 skewer
        grids. Reads ~``230400 * nbins`` float32 ~= **1.5 GB** for the real
        product (one axis only; all three would be ~4.6 GB).

    Returns
    -------
    TauGrid
        ``.tau`` is the raw ``(ngrid, ngrid, nbins)`` float32 cube (redshift
        space; LOS = last/velocity axis). ``.cube_axes`` gives the physical
        coordinate of each index (see module docstring). ``.box`` is Mpc/h.
        ``.meta`` carries ``transverse_kpc_h`` (=box/ngrid), ``los_kpc_h``
        (=box/nbins; note the cube is anisotropic), ``dv_kms`` (~10 km/s pixel
        width), ``vmax_kms``, cosmology, ``space="redshift"``.

    Raises
    ------
    ValueError
        Missing ``Header``/``tau`` dataset, or a skewer count that is not
        ``3 * ngrid**2``.
    RuntimeError
        The selected contiguous block is not actually labelled ``axis`` in
        ``spectra/axis`` (a mis-ordered / wrong-product file).
    """
    if axis not in (1, 2, 3):
        raise ValueError(f"axis must be 1, 2 or 3 (x/y/z); got {axis!r}")
    try:
        h5 = h5py.File(path, "r")
    except OSError as e:
        # A mid-transfer / truncated file is not openable by h5py.
        raise ValueError(f"{path}: cannot open (truncated or corrupt HDF5?): {e}") from e
    with h5:
        f = h5
        if "Header" not in f:
            raise ValueError(f"{path}: no 'Header' group (not a fake_spectra file?)")
        if TAU_KEY not in f:
            raise ValueError(f"{path}: no '{TAU_KEY}' dataset")
        hdr = f["Header"].attrs
        keys = set(hdr.keys())
        if "redshift" not in keys or "box" not in keys:
            raise ValueError(f"{path}: Header missing required 'redshift'/'box' attrs")
        redshift = float(hdr["redshift"])
        box_kpc_h = float(hdr["box"])
        hubble = float(hdr["hubble"]) if "hubble" in keys else None
        omegam = float(hdr["omegam"]) if "omegam" in keys else None
        omegal = float(hdr["omegal"]) if "omegal" in keys else None
        omegab = float(hdr["omegab"]) if "omegab" in keys else None
        hz = float(hdr["Hz"]) if "Hz" in keys else None

        dset = f[TAU_KEY]
        nlos = dset.shape[0]
        ngrid = int(round(math.sqrt(nlos / N_AXES)))
        if ngrid * ngrid * N_AXES != nlos:
            raise ValueError(
                f"{path}: tau has {nlos} sightlines, not 3*ngrid^2 for any integer ngrid"
            )
        per_axis = ngrid * ngrid
        sl = slice((axis - 1) * per_axis, axis * per_axis)

        # Cheap runtime guard: confirm this block really is the requested axis,
        # rather than trusting the contiguous layout blindly.
        if "spectra/axis" in f:
            axblk = f["spectra/axis"][sl]
            if not np.all(axblk == axis):
                raise RuntimeError(
                    f"{path}: skewer block for axis={axis} is not contiguously "
                    f"labelled in spectra/axis (mis-ordered or wrong product?)"
                )

        block = dset[sl]                              # (per_axis, nbins), raw float32
        nbins = block.shape[1]                        # dataset is authoritative
    tau = block.reshape(ngrid, ngrid, nbins)          # C-order; cube_axes per CUBE_AXES

    # LOS pixel velocity width; fall back to recomputing H(z) if the header lacks Hz.
    if hz is None and None not in (hubble, omegam, omegal):
        hz = units.hubble_z(redshift, omegam, omegal, hubble)
    dv_kms = None
    if hz is not None and hubble is not None:
        dv_kms = hz * (box_kpc_h / 1000.0 / hubble) / (1.0 + redshift) / nbins

    meta: Dict[str, Any] = {
        "path": str(path),
        "ngrid": ngrid,
        "nbins": nbins,
        "hubble": hubble,
        "omegam": omegam,
        "omegab": omegab,
        "omegal": omegal,
        "Hz": hz,
        "transverse_kpc_h": box_kpc_h / ngrid,
        "los_kpc_h": box_kpc_h / nbins,             # comoving LOS pixel (anisotropic!)
        "dv_kms": dv_kms,
        "vmax_kms": (dv_kms * nbins) if dv_kms is not None else None,
        "space": "redshift",                         # peculiar velocities included
        "los_is_velocity_axis": True,
    }
    return TauGrid(
        tau=tau,
        redshift=redshift,
        axis=axis,
        ngrid=ngrid,
        nbins=nbins,
        box=units.kpc_h_to_mpc_h(box_kpc_h),
        cube_axes=CUBE_AXES[axis],
        meta=meta,
    )


# --- optional derived helpers -------------------------------------------------
def to_flux(tau: np.ndarray) -> np.ndarray:
    """Transmitted flux ``F = exp(-tau)`` (same dtype as ``tau``; allocates a
    full-size copy — ~1.5 GB for a real axis cube)."""
    return np.exp(-tau)


def mean_flux(tau: np.ndarray) -> float:
    """Mean transmitted flux ``<F> = <exp(-tau)>`` (raw, unfiltered box mean of
    this axis; no DLA masking). Reduced in float64 for amplitude accuracy."""
    return float(np.exp(-tau).mean(dtype=np.float64))


def to_delta_flux(tau: np.ndarray, mean_flux_value: Optional[float] = None) -> np.ndarray:
    """Flux contrast ``delta_F = F/F_bar - 1``.

    ``mean_flux_value`` is the mean flux ``F_bar``; if ``None`` the array's own
    (float64-accumulated) mean is used. **Note:** ``b_F`` scales as ``1/F_bar``,
    so record which ``F_bar`` you used (raw box mean here vs an observational /
    tau0-rescaled target). Allocates a full-size flux copy (~1.5 GB per axis)."""
    flux = np.exp(-tau)
    mf = flux.mean(dtype=np.float64) if mean_flux_value is None else mean_flux_value
    return flux / np.float32(mf) - np.float32(1.0)
