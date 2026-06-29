"""priya_loader — ML-friendly access to the PRIYA Lyman-alpha simulation suite.

The PRIYA suite is a grid of MP-Gadget cosmological hydrodynamic simulations,
each labelled by a vector of cosmological + astrophysical/thermal parameters.
This package turns each simulation into clean 3D numpy arrays:

  * the **initial-condition (IC) density field** (from the MP-GenIC bigfile ICs,
    streamed and painted onto a mesh via :func:`priya_loader.load_ic_density`), and
  * the **Lyman-alpha optical-depth field** ``tau`` at each output redshift (the
    raw gridded ``fake_spectra`` skewers, read with h5py via
    :func:`priya_loader.load_tau_grid`). Flux ``F=exp(-tau)`` and
    ``delta_F=F/<F>-1`` are optional derived helpers.

The headline object (PLANNED, lands in a later PR) is ``PriyaDataset``, which
will loop over every simulation/redshift and yield, per snapshot, the tuple a
downstream (e.g. JAX) pipeline wants::

    [ SimParams, redshift, ic_density_3d, tau_3d ]

This release ships: :mod:`priya_loader.units`, :mod:`priya_loader.params`,
:mod:`priya_loader.runconfig`, :mod:`priya_loader.paths`, the tau loader
:mod:`priya_loader.tau`, and the IC density loader
:mod:`priya_loader.ic` (+ :mod:`priya_loader.mesh`). The ``PriyaDataset``
orchestrator that ties them into the ``[params, z, ic, tau]`` tuple lands later.

Design notes for users familiar with the simulations:
  * Paths are always explicit arguments — nothing is hard-coded. The same code
    runs on a laptop, Great Lakes, or NERSC.
  * The flux (tau) arrays are returned *exactly* as stored, only reshaped per
    skewer axis (never resampled/rebinned). The IC mesh is the thing you choose
    the resolution/orientation of, so the two cubes can be co-registered.
  * Internal MP-Gadget units (kpc/h comoving, 1e10 Msun/h, km/s with the Gadget
    sqrt(a) velocity convention) are documented in :mod:`priya_loader.units`.
"""

__version__ = "0.1.0.dev0"

from . import mesh, units  # noqa: E402
from .ic import ICField, load_ic_density  # noqa: E402
from .params import SimParams, parse_sim_name  # noqa: E402
from .paths import (  # noqa: E402
    SimulationPaths,
    discover_simulations,
    find_ic_dir,
    find_production_ic_dir,
    find_spectra_files,
)
from .runconfig import RunConfig, read_run_config  # noqa: E402
from .tau import TauGrid, load_tau_grid, mean_flux, to_delta_flux, to_flux  # noqa: E402

__all__ = [
    "__version__",
    "units",
    "mesh",
    # parameters
    "SimParams",
    "parse_sim_name",
    # initial conditions
    "ICField",
    "load_ic_density",
    # discovery
    "SimulationPaths",
    "discover_simulations",
    "find_ic_dir",
    "find_production_ic_dir",
    "find_spectra_files",
    # run config
    "RunConfig",
    "read_run_config",
    # tau / flux
    "TauGrid",
    "load_tau_grid",
    "to_flux",
    "mean_flux",
    "to_delta_flux",
]
