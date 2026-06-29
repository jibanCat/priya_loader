"""Authoritative run configuration: box size and particle grid (Ngrid).

**Why this module exists.** The per-simulation ``SimulationICs.json`` is a
*requested* config and, for a large fraction of the live suite, its ``box`` and
``npart`` were never overwritten from the SimulationRunner template — they read
``box=15, npart=192`` even for production 120 Mpc/h, 1536^3 runs. The values
that actually ran are in:

* ``mpgadget.param`` — ``InitCondFile = ICS/<box>_<Ngrid>_99`` (unambiguous), and
* ``_genic_params.ini`` — ``BoxSize`` (kpc/h) and ``Ngrid``.

We therefore take box/Ngrid from these files and treat the JSON box/npart as
untrustworthy. (``_genic_params.ini`` also contains ``NgridNu``; the parser
matches the ``Ngrid`` key exactly so it is never confused for ``NgridNu``.)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

StrPath = Union[str, os.PathLike]

# "120_1536_99" -> box (Mpc/h), Ngrid, z_init
_IC_BASENAME_RE = re.compile(r"(?P<box>\d+)_(?P<ngrid>\d+)_(?P<zinit>\d+)")


@dataclass(frozen=True)
class RunConfig:
    """The box size (Mpc/h) and production particle grid actually run."""

    box: float            # Mpc/h
    ngrid: int            # particles per side (production resolution)
    ic_basename: str      # e.g. "120_1536_99"
    source: str           # which file the values came from


def _parse_ini_value(text: str, key: str) -> Optional[str]:
    """Return the value of ``key = value`` in an ini/param file (exact key)."""
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*(.+?)\s*(?:#.*)?$", re.MULTILINE)
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def read_run_config(sim_dir: StrPath) -> RunConfig:
    """Return the authoritative :class:`RunConfig` for a simulation folder.

    Prefers ``mpgadget.param``'s ``InitCondFile`` basename; cross-checks against
    ``_genic_params.ini`` (``Ngrid``, ``BoxSize``) and raises ``ValueError`` on a
    genuine disagreement. Falls back to ``_genic_params.ini`` alone if
    ``mpgadget.param`` is absent. Raises ``FileNotFoundError`` if neither exists.
    """
    sim_dir = Path(sim_dir)
    mpgadget = sim_dir / "mpgadget.param"
    genic = sim_dir / "_genic_params.ini"

    mp_box = mp_ngrid = mp_basename = None
    if mpgadget.is_file():
        init = _parse_ini_value(mpgadget.read_text(), "InitCondFile")
        if init:
            mp_basename = Path(init).name
            m = _IC_BASENAME_RE.search(mp_basename)
            if m:
                mp_box = float(m.group("box"))
                mp_ngrid = int(m.group("ngrid"))

    gen_box = gen_ngrid = None
    if genic.is_file():
        text = genic.read_text()
        ng = _parse_ini_value(text, "Ngrid")
        bs = _parse_ini_value(text, "BoxSize")
        if ng is not None:
            gen_ngrid = int(float(ng))
        if bs is not None:
            gen_box = float(bs) / 1000.0   # kpc/h -> Mpc/h

    # Cross-check when both are available.
    if mp_ngrid is not None and gen_ngrid is not None and mp_ngrid != gen_ngrid:
        raise ValueError(
            f"{sim_dir}: Ngrid disagrees between mpgadget.param ({mp_ngrid}) "
            f"and _genic_params.ini ({gen_ngrid})"
        )

    if mp_ngrid is not None:
        box = mp_box if gen_box is None else gen_box  # genic BoxSize is exact
        return RunConfig(box=box, ngrid=mp_ngrid,
                         ic_basename=mp_basename, source="mpgadget.param")
    if gen_ngrid is not None:
        basename = f"{int(round(gen_box))}_{gen_ngrid}_99"
        return RunConfig(box=gen_box, ngrid=gen_ngrid,
                         ic_basename=basename, source="_genic_params.ini")

    raise FileNotFoundError(
        f"{sim_dir}: neither mpgadget.param nor _genic_params.ini found; "
        "cannot determine authoritative box/Ngrid"
    )
