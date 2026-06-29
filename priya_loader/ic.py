"""Initial-conditions density loader: MP-GenIC ``bigfile`` particles -> mesh.

Streams particle ``Position`` from the IC bigfile in chunks and paints them onto
a periodic ``nmesh^3`` mesh (cloud-in-cell, :mod:`priya_loader.mesh`), returning
the overdensity ``delta = rho/<rho> - 1``. The full particle load is never
resident: peak memory is the mesh plus one chunk, so this runs on a NERSC login
node for coarse meshes. ``bigfile`` is an optional dependency (``pip install
priya_loader[ic]``).

Co-registration note
--------------------
``ICField.delta`` is indexed ``[x, y, z]`` (mesh cells along the simulation x, y,
z axes; ``axes = ("x", "y", "z")``). To cross-correlate with a tau cube from
:func:`priya_loader.load_tau_grid`, align using that cube's ``cube_axes`` (e.g.
axis=1 tau is ``(y, z, LOS=x)``): transpose/orient the IC mesh to match, and use
a common ``nmesh``. The loader does the meshing only — it does not transform tau.

Units: ``Position`` is comoving kpc/h; ``box`` is reported in Mpc/h.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np

from . import mesh, units

StrPath = Union[str, os.PathLike]

#: particle-type name -> bigfile block prefix (MP-Gadget convention: 0 gas, 1 DM)
PTYPE = {"gas": 0, "dm": 1}


@dataclass
class ICField:
    """Initial-condition overdensity on a mesh."""

    delta: np.ndarray            # (nmesh, nmesh, nmesh) float32 overdensity
    nmesh: int
    box: float                   # Mpc/h
    ptype: str                   # "gas" | "dm"
    redshift: float              # IC redshift (z_init)
    npart: int                   # particles painted
    axes: tuple = ("x", "y", "z")
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)


def _attr(attrs, key):
    """Read a bigfile Header attr as a python float (bigfile stores 1-elem arrays)."""
    return float(np.asarray(attrs[key]).ravel()[0])


def load_ic_density(
    ic_dir: StrPath,
    ptype: str = "dm",
    nmesh: int = 256,
    chunk_size: int = 8_000_000,
    backend: str = "numpy",
) -> ICField:
    """Load the IC density field for one particle type as an ``nmesh^3`` overdensity.

    Parameters
    ----------
    ic_dir : str or os.PathLike
        An MP-GenIC IC bigfile directory (contains ``Header`` and ``<t>/Position``).
    ptype : {"dm", "gas"}
        Particle type (1=DM, 0=gas).
    nmesh : int
        Mesh cells per side (coarse default; raise for finer fields).
    chunk_size : int
        Particles read per streaming chunk (memory vs. overhead tradeoff).
    backend : {"numpy"}
        Painter backend. Only the lightweight numpy CIC is implemented; an
        ``nbodykit`` backend may be added later.

    Returns
    -------
    ICField
    """
    if ptype not in PTYPE:
        raise ValueError(f"ptype must be one of {sorted(PTYPE)}; got {ptype!r}")
    if backend != "numpy":
        raise ValueError(f"unknown backend {backend!r} (only 'numpy' is implemented)")
    try:
        import bigfile
    except ImportError as e:  # pragma: no cover
        raise ImportError("reading ICs needs bigfile: pip install priya_loader[ic]") from e

    bf = bigfile.File(str(ic_dir))
    attrs = bf["Header"].attrs
    box_kpc_h = _attr(attrs, "BoxSize")
    if "Redshift" in attrs.keys():
        redshift = _attr(attrs, "Redshift")
    elif "Time" in attrs.keys():
        redshift = units.scale_factor_to_redshift(_attr(attrs, "Time"))
    else:
        redshift = float("nan")

    block_name = f"{PTYPE[ptype]}/Position"
    if block_name not in bf.blocks:
        raise ValueError(
            f"{ic_dir}: no '{block_name}' block (ptype={ptype!r} absent in this IC)"
        )
    block = bf[block_name]
    npart = int(block.size)

    rho = np.zeros((nmesh, nmesh, nmesh), dtype=np.float64)
    for start in range(0, npart, chunk_size):
        pos = block[start:start + chunk_size]        # (chunk, 3) comoving kpc/h
        mesh.cic_paint(pos, nmesh, boxsize=box_kpc_h, out=rho)

    delta = mesh.to_overdensity(rho).astype(np.float32)
    return ICField(
        delta=delta,
        nmesh=nmesh,
        box=units.kpc_h_to_mpc_h(box_kpc_h),
        ptype=ptype,
        redshift=redshift,
        npart=npart,
        meta={
            "ic_dir": str(ic_dir),
            "backend": backend,
            "cell_kpc_h": box_kpc_h / nmesh,
            "hubble": _attr(attrs, "HubbleParam") if "HubbleParam" in attrs.keys() else None,
        },
    )
