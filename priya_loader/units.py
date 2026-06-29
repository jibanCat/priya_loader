"""MP-Gadget / MP-GenIC internal unit system: constants and conversions.

Getting these conventions right is the single most important correctness issue
when reading the simulations, so they live in one small, heavily-tested module.

Internal base units (the MP-Gadget defaults used by the PRIYA suite)
--------------------------------------------------------------------
* **Length**   : ``UNIT_LENGTH_IN_CM = 3.085678e21`` cm  = 1 kpc/h (comoving).
                 Positions are comoving kpc/h in ``[0, BoxSize)``; the PRIYA box
                 is ``BoxSize = 120000`` kpc/h = 120 Mpc/h.
* **Mass**     : ``UNIT_MASS_IN_G = 1.989e43`` g = 1e10 Msun/h.
* **Velocity** : ``UNIT_VELOCITY_IN_CM_S = 1e5`` cm/s = 1 km/s.

Velocity caveat (the Gadget sqrt(a) convention)
-----------------------------------------------
The velocity stored by MP-GenIC/MP-Gadget is ``u = v_peculiar / sqrt(a)`` (set
in MP-Gadget ``libgenic/zeldovich.c``). To recover the physical peculiar
velocity in km/s, multiply the stored value by ``sqrt(a)`` —
see :func:`gadget_velocity_to_peculiar_kms`.
"""
from __future__ import annotations

import numpy as np

# --- internal base units (cgs) ------------------------------------------------
UNIT_LENGTH_IN_CM = 3.085678e21    # 1 kpc/h
UNIT_MASS_IN_G = 1.989e43          # 1e10 Msun/h
UNIT_VELOCITY_IN_CM_S = 1.0e5      # 1 km/s

# --- handy derived constants --------------------------------------------------
KPC_PER_MPC = 1000.0
MPC_IN_CM = UNIT_LENGTH_IN_CM * KPC_PER_MPC   # 3.085678e24 cm = 1 Mpc/h

#: Critical density today in internal units, rho_crit,0 = 3 H0^2 / (8 pi G),
#: expressed as (1e10 Msun/h) / (Mpc/h)^3. Used for particle masses.
RHO_CRIT_1E10_MSUN_H = 27.7537


# --- length -------------------------------------------------------------------
def kpc_h_to_mpc_h(x):
    """Convert a comoving length from kpc/h to Mpc/h (e.g. 120000 -> 120).

    Scalar in -> float out; array in -> ndarray out (numpy true-division).
    """
    return x / KPC_PER_MPC


def mpc_h_to_kpc_h(x):
    """Convert a comoving length from Mpc/h to kpc/h (e.g. 120 -> 120000)."""
    return x * KPC_PER_MPC


def comoving_to_physical_mpc(x_kpc_h, scale_factor, hubble):
    """Comoving position [kpc/h] -> physical position [Mpc] (``x/1000/h * a``)."""
    return x_kpc_h / KPC_PER_MPC / hubble * scale_factor


# --- time / redshift ----------------------------------------------------------
def redshift_to_scale_factor(z):
    """a = 1 / (1 + z)."""
    return 1.0 / (1.0 + z)


def scale_factor_to_redshift(a):
    """z = 1/a - 1."""
    return 1.0 / a - 1.0


# --- Hubble rate / line-of-sight velocity scaling -----------------------------
def hubble_z(z, omega_m, omega_lambda, hubble):
    """Hubble rate H(z) in km/s/(Mpc/h) for flat LCDM (radiation neglected).

    ``H(z) = 100 h * sqrt(omega_m (1+z)^3 + omega_lambda)``. The tau loader uses
    the authoritative ``Hz`` stored in each file header when present, and this as
    a fallback to recompute it from cosmology.
    """
    return 100.0 * hubble * np.sqrt(omega_m * (1.0 + z) ** 3 + omega_lambda)


def velfac(z, omega_m, omega_lambda, hubble):
    """Comoving-distance -> peculiar-velocity factor ``H(z) / (h (1+z))``.

    Relates a comoving LOS interval to the km/s spacing of the Lyman-alpha forest
    pixels (see :mod:`priya_loader.tau` for how the pixel ``dv_kms`` is derived).
    """
    return hubble_z(z, omega_m, omega_lambda, hubble) / (hubble * (1.0 + z))


# --- velocity -----------------------------------------------------------------
def gadget_velocity_to_peculiar_kms(velocity_stored, scale_factor):
    """Physical peculiar velocity [km/s] from the stored GenIC IC velocity.

    ``v_peculiar = u_stored * sqrt(a)`` (the Gadget velocity convention,
    MP-GenIC ``libgenic/zeldovich.c``). Verified for the IC ``Velocity`` block;
    re-confirm before reusing on evolved MP-Gadget snapshots.
    """
    return velocity_stored * np.sqrt(scale_factor)


# --- mass ---------------------------------------------------------------------
def mass_to_msun(mass_internal, hubble):
    """Physical mass [Msun] from internal mass [1e10 Msun/h] (divides by h)."""
    return mass_internal * 1e10 / hubble


def particle_mass(omega, box_mpc_h, ngrid):
    """Per-particle mass [1e10 Msun/h] for one species filling the box.

    ``m = omega * rho_crit,0 * box^3 / ngrid^3`` with ``rho_crit,0`` in internal
    units. Use ``omega = Omega0 - Omegab`` for CDM and ``Omegab`` for gas. (For
    a CIC overdensity field the mass cancels; this is for absolute densities or
    gas/DM mixing.)
    """
    return omega * RHO_CRIT_1E10_MSUN_H * box_mpc_h**3 / ngrid**3
