"""Simulation parameters: the single source of truth for a PRIYA run.

Each simulation lives in a folder whose name encodes nine parameters, e.g.::

    ns0.905Ap1.79e-09herei3.97heref2.99alphaq2.1hub0.745omegamh20.145hireionz7.47bhfeedback0.043

The name is built by ``lyaemu/coarse_grid.py:build_dirname`` (lines 106-118 @
commit 27dac4f) by concatenating ``<name><value>`` with ``"%.3g"`` formatting,
in the order of ``coarse_grid.py:41`` ``param_names``
(``ns, Ap, herei, heref, alphaq, hub, omegamh2, hireionz, bhfeedback``).

Where each parameter comes from
-------------------------------
* **box size & particle grid** : the authoritative run files
  (``mpgadget.param`` / ``_genic_params.ini``) via :mod:`priya_loader.runconfig`.
  ``SimulationICs.json`` is **not** trusted for these: it is written once at
  IC-build time from a ``__dict__`` snapshot (``SimulationRunner
  simulationics.py:267-285 @ 5adf4fe``) and is not rewritten when GenIC is later
  regenerated, so for ~half the suite its ``box``/``npart`` read ``15``/``192``
  (values from the generating script) while the production run is 120 Mpc/h, 1536^3.
* **cosmology & astro/thermal params** : ``SimulationICs.json`` (high precision).
* **folder-name tokens** : used only as a cross-check (rounded; and the folder
  ``Ap`` is the primordial amplitude at PRIYA's k=0.78 Mpc^-1 pivot, deterministic-
  ally related to the JSON ``scalar_amp`` = A_s at k=0.05 Mpc^-1 by
  ``Ap = scalar_amp * (0.78/0.05)**(ns-1)`` — see :meth:`SimParams.validate`).

Parameter meanings (PRIYA 9-D Latin hypercube + fixed extras):
  cosmology   : ns, scalar_amp (A_s), hubble, omega0, omegab
  reionization/thermal : ``hireionz`` = HI reionization midpoint redshift;
                ``here_i``/``here_f``/``alpha_q`` = HeII reionization (quasar
                spectral hardening) parameters; ``heatamp`` = photo-heating
                amplitude (fixed = 1.0 across the released suite).
  feedback    : ``bhfeedback`` = black-hole/AGN feedback efficiency.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence, Union

import numpy as np

from . import runconfig

StrPath = Union[str, os.PathLike]

# Ordered keys as they appear in a folder name (concatenated "<key><float>").
_NAME_KEYS = (
    "ns", "Ap", "herei", "heref", "alphaq", "hub",
    "omegamh2", "hireionz", "bhfeedback",
)
# A float token. Every key starts with a letter NOT in this class, so the 'e' in
# "1.79e-09" cannot swallow the following "herei".
_FLOAT = r"([-+0-9.eE]+)"
# End-anchored so trailing junk (e.g. "..._v2_backup") is rejected; a leading
# mirror prefix is tolerated because we ``search`` rather than match from byte 0.
_NAME_RE = re.compile("".join(f"{k}{_FLOAT}" for k in _NAME_KEYS) + r"$")

# Pivot wavenumbers (Mpc^-1) for the amplitude cross-check.
# Origin: lya_emulator lyaemu/coarse_grid.py:154-157 @ 27dac4f --
#   conv = (0.05 / (2*pi/8))**(ns-1) ; Ap = scalar_amp(=A_s @0.05) / conv
# i.e. Ap = scalar_amp * ((2*pi/8) / 0.05)**(ns-1). The exact pivot is 2*pi/8
# (= 0.7854 Mpc^-1); the folder name's "Ap" comment rounds it to 0.78.
PIVOT_K_SCALAR_AMP = 0.05               # A_s pivot (CAMB/CLASS default)
PIVOT_K_AP = 2.0 * np.pi / 8.0          # PRIYA "Ap" pivot, 8 Mpc scale (k ~ 0.7854 Mpc^-1)

# Released-suite resolutions: npart per side (box is always 120 Mpc/h).
_ALLOWED_NPART = {1536, 3072}   # 1536 lowres, 3072 hires

# Required cosmology/astro keys in SimulationICs.json (box/npart intentionally
# excluded — those come from the run config).
_REQUIRED_JSON_KEYS = (
    "omega0", "omegab", "hubble", "scalar_amp", "ns",
    "here_i", "here_f", "alpha_q", "hireionz", "bhfeedback",
    "redshift", "redend",
)


def parse_sim_name(name: str) -> Dict[str, float]:
    """Parse the nine parameters encoded in a simulation folder name.

    Returns a dict keyed by the in-name tokens
    (``ns, Ap, herei, heref, alphaq, hub, omegamh2, hireionz, bhfeedback``).
    Raises ``ValueError`` if the name does not match (including trailing junk).
    """
    m = _NAME_RE.search(name.strip())
    if m is None:
        raise ValueError(f"not a recognised PRIYA simulation folder name: {name!r}")
    return {k: float(v) for k, v in zip(_NAME_KEYS, m.groups())}


# Convenience groupings for building ML feature vectors (public API).
COSMO_KEYS = ("omega0", "omegab", "hubble", "ns", "scalar_amp")
ASTRO_KEYS = ("here_i", "here_f", "alpha_q", "hireionz", "bhfeedback", "heatamp")


@dataclass(frozen=True)
class SimParams:
    """Canonical parameters for one simulation.

    ``box`` (Mpc/h) and ``npart`` (particles per side) come from the run config;
    cosmology/astro come from ``SimulationICs.json``.
    """

    name: str
    fidelity: str            # "lowres" (1536) | "hires" (3072)
    box: float               # Mpc/h  (authoritative, from run config)
    npart: int               # 1-D particle/grid count per side (1536 => a 1536^3 run)
    # cosmology
    omega0: float
    omegab: float
    hubble: float
    scalar_amp: float
    ns: float
    # astrophysics / thermal
    here_i: float
    here_f: float
    alpha_q: float
    hireionz: float
    bhfeedback: float
    heatamp: float
    # times: bound the full integration (z_init=99 -> z_end), distinct from the
    # science-snapshot redshift grid (~5.4..2.2) handled by the flux loader.
    z_init: float
    z_end: float
    # provenance (excluded from equality/hash so SimParams stays hashable)
    directory: Path = field(default=Path("."))
    raw: dict = field(default_factory=dict, repr=False, compare=False)
    name_params: dict = field(default_factory=dict, repr=False, compare=False)

    # --- derived (flat-LCDM) --------------------------------------------------
    @property
    def omega_lambda(self) -> float:
        """Dark-energy density, assuming a flat universe (Omega_L = 1 - Omega0).

        Verified against the genic ``OmegaLambda`` (radiation is handled
        separately by MP-Gadget, so Omega0 + Omega_L = 1 exactly).
        """
        return 1.0 - self.omega0

    @property
    def omega_m_h2(self) -> float:
        """Physical matter density Omega_m h^2 (the folder-name ``omegamh2``)."""
        return self.omega0 * self.hubble**2

    # --- construction ---------------------------------------------------------
    @classmethod
    def from_dir(cls, directory: StrPath, *, validate: bool = True) -> "SimParams":
        """Build from a simulation folder.

        Raises ``ValueError`` if ``SimulationICs.json`` is missing/malformed or a
        required key is absent, or (when ``validate``) if the parameters fail a
        consistency check.
        """
        directory = Path(directory)
        try:
            with open(directory / "SimulationICs.json", encoding="utf-8") as fh:
                raw = json.load(fh)
        except FileNotFoundError as e:
            raise ValueError(f"{directory}: SimulationICs.json not found") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"{directory}: SimulationICs.json is not valid JSON: {e}") from e

        missing = [k for k in _REQUIRED_JSON_KEYS if k not in raw]
        if missing:
            raise ValueError(f"{directory}: SimulationICs.json missing keys {missing}")

        cfg = runconfig.read_run_config(directory)   # authoritative box / Ngrid
        sp = cls(
            name=directory.name,
            fidelity=_fidelity_from_ngrid(cfg.ngrid),
            box=cfg.box,
            npart=cfg.ngrid,
            omega0=float(raw["omega0"]),
            omegab=float(raw["omegab"]),
            hubble=float(raw["hubble"]),
            scalar_amp=float(raw["scalar_amp"]),
            ns=float(raw["ns"]),
            here_i=float(raw["here_i"]),
            here_f=float(raw["here_f"]),
            alpha_q=float(raw["alpha_q"]),
            hireionz=float(raw["hireionz"]),
            bhfeedback=float(raw["bhfeedback"]),
            heatamp=float(raw.get("heatamp", 1.0)),
            z_init=float(raw["redshift"]),
            z_end=float(raw["redend"]),
            directory=directory,
            raw=raw,
            name_params=parse_sim_name(directory.name),
        )
        if validate:
            sp.validate()
        return sp

    # --- validation -----------------------------------------------------------
    def validate(self, rtol: float = 0.02, amp_rtol: float = 0.05) -> None:
        """Consistency checks; raise ``ValueError`` on any gross mismatch.

        1. Resolution invariant: ``(npart, box)`` must be one of the released
           PRIYA combinations (1536/120 lowres, 3072/120 hires).
        2. Folder-name cosmology/astro tokens must match the JSON within ``rtol``.
        3. Amplitude cross-check: the folder ``Ap`` must match the pivot-rescaled
           JSON ``scalar_amp`` within ``amp_rtol`` (catches a JSON/folder
           amplitude mispairing — the only otherwise-unvalidated cosmo param).
        """
        if self.npart not in _ALLOWED_NPART or not np.isclose(self.box, 120.0, atol=1e-3):
            raise ValueError(
                f"{self.name}: resolution (npart={self.npart}, box={self.box}) is "
                f"not a released PRIYA combination (npart in {sorted(_ALLOWED_NPART)}, box=120)"
            )
        token_checks = {
            "ns": self.ns, "hub": self.hubble, "herei": self.here_i,
            "heref": self.here_f, "alphaq": self.alpha_q,
            "hireionz": self.hireionz, "bhfeedback": self.bhfeedback,
            "omegamh2": self.omega_m_h2,
        }
        for key, canonical in token_checks.items():
            named = self.name_params.get(key)
            if named is None:
                continue
            if not np.isclose(named, canonical, rtol=rtol, atol=1e-12):
                raise ValueError(
                    f"{self.name}: folder token {key}={named} disagrees with "
                    f"JSON-derived {canonical} (rtol={rtol})"
                )
        ap_named = self.name_params.get("Ap")
        if ap_named is not None:
            ap_pred = self.scalar_amp * (PIVOT_K_AP / PIVOT_K_SCALAR_AMP) ** (self.ns - 1.0)
            # atol=0: values are ~1e-9, so np.isclose's default atol=1e-8 would
            # otherwise mask any real disagreement.
            if not np.isclose(ap_named, ap_pred, rtol=amp_rtol, atol=0.0):
                raise ValueError(
                    f"{self.name}: folder Ap={ap_named:.3e} disagrees with pivot-"
                    f"rescaled scalar_amp={ap_pred:.3e} (amp_rtol={amp_rtol})"
                )

    def as_vector(self, keys: Sequence[str]) -> np.ndarray:
        """Return the named parameters as a float array, in the given order."""
        return np.array([getattr(self, k) for k in keys], dtype=float)

    def to_dict(self) -> dict:
        """Resolved parameters as a JSON-serialisable dict (authoritative box/npart
        from the run config, not the possibly-stale JSON)."""
        return {
            "name": self.name, "fidelity": self.fidelity,
            "box": self.box, "npart": self.npart,
            "omega0": self.omega0, "omegab": self.omegab, "hubble": self.hubble,
            "omega_lambda": self.omega_lambda, "omega_m_h2": self.omega_m_h2,
            "scalar_amp": self.scalar_amp, "ns": self.ns,
            "here_i": self.here_i, "here_f": self.here_f, "alpha_q": self.alpha_q,
            "hireionz": self.hireionz, "bhfeedback": self.bhfeedback,
            "heatamp": self.heatamp, "z_init": self.z_init, "z_end": self.z_end,
        }


def _fidelity_from_ngrid(ngrid: int) -> str:
    if ngrid == 1536:
        return "lowres"
    if ngrid == 3072:
        return "hires"
    return f"npart{ngrid}"
