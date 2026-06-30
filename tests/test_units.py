"""Physics-unit tests for the MP-Gadget internal unit system.

These guard the single most error-prone part of reading the simulations: the
internal unit conventions (kpc/h comoving, 1e10 Msun/h, km/s with the Gadget
sqrt(a) peculiar-velocity convention). If any of these break, every downstream
array is silently wrong.
"""
import math

import numpy as np
import pytest

from priya_loader import units


def test_unit_constants_are_canonical():
    # MP-Gadget internal base units (cgs).
    assert units.UNIT_LENGTH_IN_CM == pytest.approx(3.085678e21)   # 1 kpc/h
    assert units.UNIT_MASS_IN_G == pytest.approx(1.989e43)         # 1e10 Msun/h
    assert units.UNIT_VELOCITY_IN_CM_S == pytest.approx(1.0e5)     # 1 km/s


def test_one_mpc_is_a_thousand_kpc_in_cm():
    # 1 Mpc/h = 1000 kpc/h; the cm value must be consistent.
    assert units.MPC_IN_CM == pytest.approx(units.UNIT_LENGTH_IN_CM * 1000.0)
    assert units.MPC_IN_CM == pytest.approx(3.085678e24)


def test_box_kpc_h_to_mpc_h():
    # The PRIYA box is BoxSize=120000 kpc/h == 120 Mpc/h.
    assert units.kpc_h_to_mpc_h(120000.0) == pytest.approx(120.0)
    assert units.mpc_h_to_kpc_h(120.0) == pytest.approx(120000.0)


def test_kpc_mpc_roundtrip_array():
    x = np.array([0.0, 250.0, 120000.0])
    out = units.mpc_h_to_kpc_h(units.kpc_h_to_mpc_h(x))
    np.testing.assert_allclose(out, x)


def test_scale_factor_redshift_roundtrip():
    assert units.redshift_to_scale_factor(99.0) == pytest.approx(0.01)
    assert units.scale_factor_to_redshift(0.01) == pytest.approx(99.0)


def test_a_times_one_plus_z_is_one():
    for z in (2.0, 2.8, 5.4, 99.0):
        a = units.redshift_to_scale_factor(z)
        assert a * (1.0 + z) == pytest.approx(1.0)


def test_gadget_velocity_uses_sqrt_a_convention():
    # Stored GenIC velocity u relates to peculiar velocity by v_pec = u * sqrt(a).
    # At a = 0.25 the factor is exactly 0.5.
    assert units.gadget_velocity_to_peculiar_kms(10.0, 0.25) == pytest.approx(5.0)
    a = 0.3125
    np.testing.assert_allclose(
        units.gadget_velocity_to_peculiar_kms(np.array([2.0, 4.0]), a),
        np.array([2.0, 4.0]) * math.sqrt(a),
    )


def test_mass_to_msun():
    # internal mass unit is 1e10 Msun/h -> physical Msun divides by h.
    assert units.mass_to_msun(1.0, hubble=0.7) == pytest.approx(1e10 / 0.7)
    assert units.mass_to_msun(2.0, hubble=0.745) == pytest.approx(2e10 / 0.745)


def test_hubble_z_matches_reference_value():
    # Verified ground-truth (ref 02): Hz(z=5, Om=0.32372, OL=0.67628, h=0.65833) ~ 553.15 km/s/(Mpc/h)
    hz = units.hubble_z(5.0, omega_m=0.3237173529883032,
                        omega_lambda=0.6762826470116968, hubble=0.6583333333333333)
    assert hz == pytest.approx(553.15, rel=2e-3)


def test_hubble_z_at_z0_is_100h_for_flat():
    assert units.hubble_z(0.0, omega_m=0.3, omega_lambda=0.7, hubble=0.7) == pytest.approx(70.0)


def test_velfac_is_hubble_z_over_h_one_plus_z():
    args = dict(omega_m=0.3237, omega_lambda=0.6763, hubble=0.6583)
    z = 5.0
    expected = units.hubble_z(z, **args) / (args["hubble"] * (1.0 + z))
    assert units.velfac(z, **args) == pytest.approx(expected)


def test_comoving_to_physical_mpc():
    # 120000 kpc/h comoving, a=0.25, h=0.7 -> 120/0.7*0.25 Mpc physical
    out = units.comoving_to_physical_mpc(120000.0, scale_factor=0.25, hubble=0.7)
    assert out == pytest.approx(120.0 / 0.7 * 0.25)


def test_particle_mass_matches_reference_512():
    # ref 03 §4.6: for the 120/512 companion, M_dm ~ 0.0792, M_gas ~ 0.0144 (1e10 Msun/h)
    omega0, omegab = 0.26206025134002975, 0.04035854240799964
    m_dm = units.particle_mass(omega0 - omegab, box_mpc_h=120.0, ngrid=512)
    m_gas = units.particle_mass(omegab, box_mpc_h=120.0, ngrid=512)
    assert m_dm == pytest.approx(0.0792, rel=1e-2)
    assert m_gas == pytest.approx(0.0144, rel=1e-2)


def test_growth_factor_normalized_and_decreasing():
    om, ol = 0.3, 0.7
    assert units.growth_factor(0.0, om, ol) == pytest.approx(1.0)
    assert units.growth_factor(2.0, om, ol) < units.growth_factor(1.0, om, ol) < 1.0


def test_growth_factor_matter_domination_scales_as_a():
    # deep in matter domination D ∝ a, so D(99)/D(49) ≈ (1+49)/(1+99) = 0.5
    om, ol = 0.3, 0.7
    ratio = units.growth_factor(99.0, om, ol) / units.growth_factor(49.0, om, ol)
    assert ratio == pytest.approx(50.0 / 100.0, rel=0.02)


def test_growth_factor_ratio_zflux_to_zinit():
    # the b_F amplitude factor D(z_flux)/D(z_init): ~26x at z=2.8 vs z=99 (LCDM)
    om, ol = 0.3, 0.7
    ratio = units.growth_factor(2.8, om, ol) / units.growth_factor(99.0, om, ol)
    assert ratio == pytest.approx(26.0, rel=0.1)


def test_growth_rate_approx():
    om, ol = 0.3, 0.7
    assert units.growth_rate(0.0, om, ol) == pytest.approx(0.3 ** 0.55, rel=1e-6)
    assert units.growth_rate(5.0, om, ol) > 0.95            # -> 1 in matter domination
