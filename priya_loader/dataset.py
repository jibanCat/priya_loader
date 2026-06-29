"""``PriyaDataset`` — the headline orchestrator.

Loops over every simulation and output redshift in a PRIYA tree and yields, per
snapshot, the tuple a downstream (e.g. JAX) pipeline wants::

    Sample(params, redshift, ic, tau)

where ``ic`` is the initial-condition CIC overdensity mesh (z_init, real space)
and ``tau`` is the raw Lyman-alpha optical-depth cube for one LOS axis. It ties
together :mod:`priya_loader.paths`, :mod:`priya_loader.params`,
:mod:`priya_loader.ic`, and :mod:`priya_loader.tau`.

Design:
  * **Lazy** — ``iter_samples()`` is a generator; only one (sim, z) is resident
    at a time, so it runs on a NERSC login node (raise ``ic_nmesh`` carefully).
  * **IC loaded once per simulation** and shared across that sim's redshifts (the
    IC is z-independent; the consumer growth-scales per redshift).
  * **Graceful** on partial/mid-transfer data: a sim with no staged tau is
    skipped; a missing production IC yields ``ic=None`` (tau-only samples); a
    folder that fails parameter validation is skipped with a warning.
  * **Co-registration / growth** are the consumer's job — see
    :mod:`priya_loader.ic` and :mod:`priya_loader.tau`. This object only loads.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np

from . import paths
from .ic import load_ic_density
from .params import SimParams
from .tau import load_tau_grid

StrPath = Union[str, os.PathLike]


@dataclass
class Sample:
    """One (simulation, redshift) record: params + IC mesh + tau cube."""

    params: SimParams
    redshift: float
    ic: Optional[np.ndarray]      # (nmesh, nmesh, nmesh) float32, or None if absent
    tau: Optional[np.ndarray]     # (ngrid, ngrid, nbins) float32, or None
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    def as_tuple(self):
        """Return ``(params, redshift, ic, tau)`` (Roger's requested shape)."""
        return (self.params, self.redshift, self.ic, self.tau)


class PriyaDataset:
    """Iterable dataset over a PRIYA simulation tree.

    Parameters
    ----------
    root : str or os.PathLike
        Directory containing simulation folders (e.g. ``.../emu_full``).
    fidelity : {"lowres", "hires", None}
        Keep only simulations of this fidelity (None = all).
    ic_nmesh : int
        Mesh size for the IC density field.
    ic_ptype : {"dm", "gas"}
        IC particle type to paint.
    flux_axis : {1, 2, 3}
        Line-of-sight axis for the tau cube.
    load_ic : bool
        If False, skip IC loading (``Sample.ic`` is None) — fast, tau-only.
    validate : bool
        Validate each ``SimParams`` (resolution + name/JSON consistency).
    """

    def __init__(
        self,
        root: StrPath,
        *,
        fidelity: Optional[str] = None,
        ic_nmesh: int = 256,
        ic_ptype: str = "dm",
        flux_axis: int = 1,
        load_ic: bool = True,
        validate: bool = True,
    ):
        self.root = Path(root)
        self.fidelity = fidelity
        self.ic_nmesh = ic_nmesh
        self.ic_ptype = ic_ptype
        self.flux_axis = flux_axis
        self.load_ic = load_ic
        self.validate = validate

    def __iter__(self) -> Iterator[Sample]:
        return self.iter_samples()

    def iter_samples(self) -> Iterator[Sample]:
        """Yield one :class:`Sample` per (simulation, redshift)."""
        for sim in paths.discover_simulations(self.root):
            try:
                params = SimParams.from_dir(sim.directory, validate=self.validate)
            except ValueError as e:
                warnings.warn(f"skipping {sim.name}: {e}")
                continue
            if self.fidelity is not None and params.fidelity != self.fidelity:
                continue
            tau_files = paths.find_spectra_files(sim.directory)
            if not tau_files:
                continue

            ic_arr, ic_meta = None, {}
            if self.load_ic:
                ic_dir = paths.find_production_ic_dir(sim.directory)
                if ic_dir is not None:
                    f = load_ic_density(ic_dir, ptype=self.ic_ptype, nmesh=self.ic_nmesh)
                    ic_arr = f.delta
                    ic_meta = {"ic_axes": f.axes, "ic_space": f.space,
                               "ic_redshift": f.redshift, **f.meta}

            for snap, tau_path in tau_files:
                g = load_tau_grid(tau_path, axis=self.flux_axis)
                yield Sample(
                    params=params,
                    redshift=g.redshift,
                    ic=ic_arr,
                    tau=g.tau,
                    meta={
                        "sim": sim.name,
                        "snap": snap,
                        "flux_axis": self.flux_axis,
                        "tau_cube_axes": g.cube_axes,
                        "tau_space": g.meta.get("space"),
                        "tau_meta": g.meta,
                        "ic_meta": ic_meta,
                    },
                )

    def to_list(self) -> List[Sample]:
        """Eagerly materialise all samples (memory-heavy — prefer iteration)."""
        return list(self.iter_samples())

    def export(self, outdir: StrPath, fmt: str = "npz") -> List[Path]:
        """Write one ``.npz`` per (sim, z) with ``ic``, ``tau``, ``redshift``,
        and JSON-serialised ``params``. Returns the written paths.

        NERSC-login-safe: streams one sample at a time. (The IC is repeated per
        redshift on disk; pass ``load_ic=False`` + export IC separately if that
        duplication matters.)
        """
        if fmt != "npz":
            raise ValueError(f"unknown export format {fmt!r} (only 'npz')")
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        for s in self.iter_samples():
            fname = f"{s.meta['sim']}__z{s.redshift:.2f}__axis{self.flux_axis}.npz"
            path = outdir / fname
            np.savez_compressed(
                path,
                ic=s.ic if s.ic is not None else np.array([]),
                tau=s.tau if s.tau is not None else np.array([]),
                redshift=np.float64(s.redshift),
                params=json.dumps(s.params.raw),
                meta=json.dumps({k: v for k, v in s.meta.items()
                                 if k in ("sim", "snap", "flux_axis")}),
            )
            written.append(path)
        return written
