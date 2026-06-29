"""priya_loader — ML-friendly access to the PRIYA Lyman-alpha simulation suite.

The PRIYA suite is a grid of MP-Gadget cosmological hydrodynamic simulations,
each labelled by a vector of cosmological + astrophysical/thermal parameters.
This package turns each simulation into clean 3D numpy arrays:

  * the **initial-condition (IC) density field** (from the MP-GenIC bigfile ICs,
    painted onto a mesh), and
  * the **Lyman-alpha flux field** ``delta_F`` at each output redshift (from the
    pre-computed gridded ``fake_spectra`` optical-depth files, read with h5py).

The headline object (PLANNED, lands in a later PR) is ``PriyaDataset``, which
will loop over every simulation/redshift and yield, per snapshot, the tuple a
downstream (e.g. JAX) pipeline wants::

    [ SimParams, redshift, ic_density_3d, flux_3d ]

This foundation release ships the building blocks: :mod:`priya_loader.units`,
:mod:`priya_loader.params`, :mod:`priya_loader.runconfig`, and
:mod:`priya_loader.paths`.

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

__all__ = ["__version__"]
