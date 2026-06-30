"""Tests for the initial-conditions density loader (bigfile -> CIC mesh).

Local IC bigfiles are empty skeletons (mid-transfer), so these build a tiny
SYNTHETIC bigfile with the real block/Header layout and exercise the loader
end-to-end. Requires the optional ``bigfile`` dependency.
"""
import math

import numpy as np
import pytest

bigfile = pytest.importorskip("bigfile")

from priya_loader import ic, mesh


def _make_ic_bigfile(path, ngrid=8, box=120000.0, ptypes=(1,), redshift=99.0,
                     time=None, clustered=False, with_boxsize=True, positions=None):
    """Write a tiny IC bigfile. Particles on a regular ngrid^3 lattice at cell
    centres (=> uniform density), unless ``clustered`` or explicit ``positions``."""
    if positions is not None:
        pos = np.asarray(positions, "f8")
    else:
        g = (np.arange(ngrid) + 0.5) * box / ngrid
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype("f8")
        if clustered:
            pos = np.full_like(pos, box / 2.0)
    with bigfile.File(str(path), create=True) as bf:
        for t in ptypes:
            bf.create_from_array(f"{t}/Position", pos)
        bf.create("Header")
        h = bf["Header"]
        if with_boxsize:
            h.attrs["BoxSize"] = box
        if redshift is not None:
            h.attrs["Redshift"] = redshift
        if time is not None:
            h.attrs["Time"] = time
        h.attrs["HubbleParam"] = 0.7
        h.attrs["Omega0"] = 0.3
        h.attrs["OmegaLambda"] = 0.7
    return path


def test_uniform_lattice_gives_zero_overdensity(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8)
    field = ic.load_ic_density(p, ptype="dm", nmesh=8)
    assert field.delta.shape == (8, 8, 8)
    np.testing.assert_allclose(field.delta, 0.0, atol=1e-9)


def test_metadata(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, box=120000.0, redshift=99.0)
    field = ic.load_ic_density(p, ptype="dm", nmesh=8)
    assert field.box == pytest.approx(120.0)       # Mpc/h
    assert field.redshift == pytest.approx(99.0)
    assert field.ptype == "dm"
    assert field.nmesh == 8
    assert field.npart == 8 ** 3
    assert field.delta.dtype == np.float32
    assert field.axes == ("x", "y", "z")


def test_real_space_label(tmp_path):
    # IC delta is real-space (symmetric to tau's space="redshift").
    p = _make_ic_bigfile(tmp_path / "ic")
    field = ic.load_ic_density(p, ptype="dm", nmesh=8)
    assert field.space == "real"
    assert field.meta["los_is_velocity_axis"] is False


def test_meta_carries_cosmology_for_growth(tmp_path):
    # Omega0/OmegaLambda must be exposed so the consumer can compute D(z).
    p = _make_ic_bigfile(tmp_path / "ic")
    m = ic.load_ic_density(p, ptype="dm", nmesh=8).meta
    assert m["Omega0"] == pytest.approx(0.3)
    assert m["OmegaLambda"] == pytest.approx(0.7)
    assert m["hubble"] == pytest.approx(0.7)


def test_clustered_gives_structure(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, clustered=True)
    field = ic.load_ic_density(p, ptype="dm", nmesh=8)
    assert field.delta.max() > 10
    assert field.delta.min() == pytest.approx(-1.0)
    assert field.delta.mean() == pytest.approx(0.0, abs=1e-6)


def test_ptype_selection_gas_vs_dm(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, ptypes=(0, 1))
    assert ic.load_ic_density(p, ptype="gas", nmesh=8).ptype == "gas"
    assert ic.load_ic_density(p, ptype="dm", nmesh=8).ptype == "dm"


def test_mass_cancels_gas_equals_dm(tmp_path):
    # Same positions painted as gas vs dm give identical delta (mass cancels).
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, ptypes=(0, 1), clustered=True)
    dm = ic.load_ic_density(p, ptype="dm", nmesh=8).delta
    gas = ic.load_ic_density(p, ptype="gas", nmesh=8).delta
    np.testing.assert_allclose(dm, gas, rtol=1e-12)


def test_missing_ptype_raises(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, ptypes=(1,))   # dm only
    with pytest.raises(ValueError):
        ic.load_ic_density(p, ptype="gas", nmesh=8)


def test_invalid_ptype_name_raises(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8)
    with pytest.raises(ValueError):
        ic.load_ic_density(p, ptype="stars", nmesh=8)


def test_missing_boxsize_raises(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, with_boxsize=False)
    with pytest.raises(ValueError):
        ic.load_ic_density(p, ptype="dm", nmesh=8)


def test_redshift_from_time_when_no_redshift(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, redshift=None, time=0.01)
    assert ic.load_ic_density(p, ptype="dm", nmesh=8).redshift == pytest.approx(99.0)


def test_redshift_nan_and_warns_when_neither(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, redshift=None, time=None)
    with pytest.warns(Warning):
        field = ic.load_ic_density(p, ptype="dm", nmesh=8)
    assert math.isnan(field.redshift)


def test_origin_alignment_single_particle(tmp_path):
    # A particle at comoving (j,k,l)*box/nmesh lands fully in IC node [j,k,l]
    # (origin 0), matching the fake_spectra cofm grid. Locks co-registration.
    box = 120000.0
    H = box / 8
    p = _make_ic_bigfile(tmp_path / "ic", positions=[[2 * H, 3 * H, 5 * H]])
    rho = mesh.cic_paint(np.array([[2 * H, 3 * H, 5 * H]]), nmesh=8, boxsize=box)
    assert rho[2, 3, 5] == pytest.approx(1.0)
    _ = p


def test_chunked_matches_unchunked(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, clustered=False)
    full = ic.load_ic_density(p, ptype="dm", nmesh=8, chunk_size=10 ** 9)
    chunked = ic.load_ic_density(p, ptype="dm", nmesh=8, chunk_size=37)
    np.testing.assert_allclose(full.delta, chunked.delta, rtol=1e-6)


def test_invalid_nmesh_raises(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8)
    with pytest.raises(ValueError):
        ic.load_ic_density(p, ptype="dm", nmesh=0)
