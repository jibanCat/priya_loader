"""Tests for the initial-conditions density loader (bigfile -> CIC mesh).

Local IC bigfiles are empty skeletons (mid-transfer), so these build a tiny
SYNTHETIC bigfile with the real block/Header layout and exercise the loader
end-to-end. Requires the optional ``bigfile`` dependency.
"""
import numpy as np
import pytest

bigfile = pytest.importorskip("bigfile")

from priya_loader import ic, mesh


def _make_ic_bigfile(path, ngrid=8, box=120000.0, ptypes=(1,), redshift=99.0,
                     clustered=False):
    """Write a tiny IC bigfile. Particles on a regular ngrid^3 lattice at cell
    centres (=> uniform density), unless ``clustered``."""
    g = (np.arange(ngrid) + 0.5) * box / ngrid
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype("f8")
    if clustered:
        pos = np.full_like(pos, box / 2.0)        # all particles at the centre
    with bigfile.File(str(path), create=True) as bf:
        for t in ptypes:
            bf.create_from_array(f"{t}/Position", pos)
        bf.create("Header")
        bf["Header"].attrs["BoxSize"] = box
        bf["Header"].attrs["Redshift"] = redshift
        bf["Header"].attrs["HubbleParam"] = 0.7
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


def test_clustered_gives_structure(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, clustered=True)
    field = ic.load_ic_density(p, ptype="dm", nmesh=8)
    assert field.delta.max() > 10
    assert field.delta.min() == pytest.approx(-1.0)


def test_ptype_selection_gas_vs_dm(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, ptypes=(0, 1))
    assert ic.load_ic_density(p, ptype="gas", nmesh=8).ptype == "gas"
    assert ic.load_ic_density(p, ptype="dm", nmesh=8).ptype == "dm"


def test_missing_ptype_raises(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, ptypes=(1,))   # dm only
    with pytest.raises(ValueError):
        ic.load_ic_density(p, ptype="gas", nmesh=8)


def test_chunked_matches_unchunked(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, clustered=False)
    full = ic.load_ic_density(p, ptype="dm", nmesh=8, chunk_size=10 ** 9)
    chunked = ic.load_ic_density(p, ptype="dm", nmesh=8, chunk_size=37)
    np.testing.assert_allclose(full.delta, chunked.delta, rtol=1e-6)


def test_invalid_ptype_name_raises(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8)
    with pytest.raises(ValueError):
        ic.load_ic_density(p, ptype="stars", nmesh=8)
