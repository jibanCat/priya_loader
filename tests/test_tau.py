"""Tests for the tau (Lyman-alpha optical depth) loader.

The loader returns the RAW tau skewers reshaped per line-of-sight axis to a
(ngrid, ngrid, nbins) cube — never modified/resampled. Flux F=exp(-tau) and
delta_flux are optional derived helpers.

Fixtures build a tiny grid_480-schema HDF5 (ngrid=4, 3 axes) so reshape/axis
selection can be checked exactly, without the real ~3 GB files.
"""
import os

import h5py
import numpy as np
import pytest

from priya_loader import paths
from priya_loader import tau as T


def _make_tau_h5(path, ng=4, nbins=5, redshift=4.2, box=120000.0):
    """Write a minimal grid_480-schema file. tau[g,k] = g*10 + k (g over all
    3*ng*ng skewers, axis blocks contiguous in order 1,2,3)."""
    per_axis = ng * ng
    nlos = 3 * per_axis
    g = np.arange(nlos)[:, None]
    k = np.arange(nbins)[None, :]
    data = (g * 10 + k).astype(np.float32)
    with h5py.File(path, "w") as f:
        grp = f.create_group("tau/H/1")
        grp.create_dataset("1215", data=data)
        hdr = f.create_group("Header")
        hdr.attrs["redshift"] = redshift
        hdr.attrs["nbins"] = nbins
        hdr.attrs["box"] = box
        hdr.attrs["hubble"] = 0.7
        hdr.attrs["omegam"] = 0.3
        hdr.attrs["omegab"] = 0.05
        hdr.attrs["omegal"] = 0.7
        hdr.attrs["Hz"] = 500.0
    return path


@pytest.fixture
def tau_file(tmp_path):
    return _make_tau_h5(tmp_path / "lya_forest_spectra_grid_480.hdf5")


def test_load_tau_grid_shape(tau_file):
    g = T.load_tau_grid(tau_file, axis=1)
    assert g.tau.shape == (4, 4, 5)
    assert g.ngrid == 4 and g.nbins == 5
    assert g.axis == 1


def test_reshape_is_c_order(tau_file):
    # axis 1 block is rows [0:16); cube[a,b,k] == (a*4+b)*10 + k
    g = T.load_tau_grid(tau_file, axis=1)
    assert g.tau[0, 0, 0] == pytest.approx(0.0)
    assert g.tau[0, 1, 0] == pytest.approx(10.0)     # row 1
    assert g.tau[1, 0, 0] == pytest.approx(40.0)     # row 4
    assert g.tau[2, 3, 4] == pytest.approx((2 * 4 + 3) * 10 + 4)


def test_axis_selection_picks_correct_block(tau_file):
    # axis 2 block is rows [16:32); cube[0,0,0] == 16*10 + 0
    g2 = T.load_tau_grid(tau_file, axis=2)
    assert g2.tau[0, 0, 0] == pytest.approx(160.0)
    g3 = T.load_tau_grid(tau_file, axis=3)
    assert g3.tau[0, 0, 0] == pytest.approx(320.0)   # row 32


def test_returns_raw_tau_not_flux(tau_file):
    g = T.load_tau_grid(tau_file, axis=1)
    # raw tau, NOT exp(-tau): the (2,3,4) entry is 114.0, not exp(-114)~0
    assert g.tau[2, 3, 4] == pytest.approx(114.0)
    assert g.tau.dtype == np.float32


def test_redshift_and_meta_from_header(tau_file):
    g = T.load_tau_grid(tau_file, axis=1)
    assert g.redshift == pytest.approx(4.2)
    assert g.box == pytest.approx(120.0)             # Mpc/h (from box=120000 kpc/h)
    assert g.meta["hubble"] == pytest.approx(0.7)
    assert g.meta["omegam"] == pytest.approx(0.3)


def test_invalid_axis_raises(tau_file):
    with pytest.raises(ValueError):
        T.load_tau_grid(tau_file, axis=4)


def test_flux_helpers(tau_file):
    g = T.load_tau_grid(tau_file, axis=1)
    flux = T.to_flux(g.tau)
    np.testing.assert_allclose(flux, np.exp(-g.tau), rtol=1e-6)
    assert T.mean_flux(g.tau) == pytest.approx(float(np.exp(-g.tau).mean()))
    df = T.to_delta_flux(g.tau, mean_flux=0.5)
    np.testing.assert_allclose(df, np.exp(-g.tau) / 0.5 - 1.0, rtol=1e-6)


@pytest.mark.realdata
def test_load_real_grid_480():
    """Load a genuine grid_480 file (auto-skipped unless PRIYA_DATA_ROOT is set)."""
    root = os.environ["PRIYA_DATA_ROOT"]
    sims = paths.discover_simulations(root)
    found = next(((s, fs) for s in sims if (fs := paths.find_spectra_files(s.directory))), None)
    if found is None:
        pytest.skip("no staged grid_480 tau files under PRIYA_DATA_ROOT")
    _, fs = found
    g = T.load_tau_grid(fs[0][1], axis=1)
    assert g.ngrid == 480
    assert g.tau.shape == (480, 480, g.nbins)
    assert g.box == pytest.approx(120.0)
    assert g.meta["transverse_kpc_h"] == pytest.approx(250.0)
    assert g.meta["dv_kms"] == pytest.approx(10.0, abs=1.0)   # ~10 km/s pixels
    assert 0.0 < T.mean_flux(g.tau) < 1.0
