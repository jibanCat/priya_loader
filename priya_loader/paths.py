"""Locate simulations and their IC / flux files on disk.

Everything here takes an explicit root or simulation directory — **no path is
ever hard-coded**, so the same code runs on Great Lakes, NERSC, or a laptop.
All helpers are robust to *partial* data (the suite is often mid-transfer): a
missing IC grid or an empty SPECTRA directory yields ``None`` / an empty list
rather than an error.

Layout reminder (one simulation folder)::

    <sim>/
      SimulationICs.json                     # marks a simulation folder
      ICS/<box>_<Ngrid>_99/                  # MP-GenIC bigfile ICs (production = Ngrid)
      output/SPECTRA_<NNN>/<grid flux file>  # one gridded tau file per redshift
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

from . import runconfig

StrPath = Union[str, os.PathLike]

#: Filename of the gridded fake_spectra optical-depth (tau) product we read
#: (the 480^2-per-axis GriddedSpectra save file used by the PRIYA pipeline).
GRID_SPECTRA_FILENAME = "lya_forest_spectra_grid_480.hdf5"

_SIM_MARKER = "SimulationICs.json"
# IC bigfile dir naming "<box_Mpc>_<Ngrid>_<z_init>" (e.g. 120_1536_99): the
# GenIC FileBase set by SimulationRunner simulationics.py:210-211 @ 5adf4fe.
_IC_GRID_RE = re.compile(r"^(?P<box>\d+)_(?P<ngrid>\d+)_(?P<zinit>\d+)$")
_SPECTRA_RE = re.compile(r"^SPECTRA_(\d+)$")


@dataclass(frozen=True)
class SimulationPaths:
    """A discovered simulation: its folder name and absolute directory."""

    name: str
    directory: Path


def discover_simulations(root: StrPath) -> List[SimulationPaths]:
    """Return all simulation folders under ``root``, sorted by name.

    A directory is a simulation iff it contains ``SimulationICs.json``; this
    cleanly skips emulator products, scripts, and other non-simulation entries
    that share the parent directory.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    sims = [
        SimulationPaths(name=child.name, directory=child)
        for child in root.iterdir()
        if child.is_dir() and (child / _SIM_MARKER).is_file()
    ]
    return sorted(sims, key=lambda s: s.name)


def find_ic_dir(sim_dir: StrPath, ngrid: Optional[int] = None) -> Optional[Path]:
    """Return an IC bigfile directory that is present on disk, or ``None``.

    With ``ngrid`` given, return exactly ``ICS/<box>_<ngrid>_99`` if it exists,
    else ``None``. With ``ngrid=None`` (default), return the **largest IC grid
    currently staged**.

    .. warning::
       "Largest staged" is *not* necessarily the production resolution — on a
       partially-transferred tree only the low-res ``120_512_99`` companion may
       be present. To get the production IC (and ``None`` if it is not yet
       staged), use :func:`find_production_ic_dir`.
    """
    ics = Path(sim_dir) / "ICS"
    if not ics.is_dir():
        return None
    grids: List[Tuple[int, Path]] = []
    for child in ics.iterdir():
        if not child.is_dir():
            continue
        m = _IC_GRID_RE.match(child.name)
        if m:
            grids.append((int(m.group("ngrid")), child))
    if not grids:
        return None
    if ngrid is not None:
        for g, path in grids:
            if g == ngrid:
                return path
        return None
    return max(grids, key=lambda gp: gp[0])[1]


def find_production_ic_dir(sim_dir: StrPath) -> Optional[Path]:
    """Return the *production-resolution* IC dir if staged, else ``None``.

    The production grid is read from the authoritative run config
    (``mpgadget.param`` / ``_genic_params.ini``), so a staged low-res companion
    is never mistaken for it. ``None`` means the production IC has not been
    transferred yet (a common mid-transfer state) — callers can then decide
    whether to fall back to the companion via ``find_ic_dir(sim_dir, 512)``.
    """
    cfg = runconfig.read_run_config(sim_dir)
    return find_ic_dir(sim_dir, ngrid=cfg.ngrid)


def find_spectra_files(
    sim_dir: StrPath, filename: str = GRID_SPECTRA_FILENAME
) -> List[Tuple[int, Path]]:
    """Return ``[(snap_index, path), ...]`` for present, non-empty flux files.

    Sorted by snapshot index. Empty (0-byte) files and SPECTRA directories that
    do not yet contain the gridded flux file are skipped — exactly the
    mid-transfer states seen in the live data.
    """
    out = Path(sim_dir) / "output"
    if not out.is_dir():
        return []
    found: List[Tuple[int, Path]] = []
    for child in out.iterdir():
        if not child.is_dir():
            continue
        m = _SPECTRA_RE.match(child.name)
        if not m:
            continue
        f = child / filename
        if f.is_file() and f.stat().st_size > 0:
            found.append((int(m.group(1)), f))
    return sorted(found, key=lambda sp: sp[0])
