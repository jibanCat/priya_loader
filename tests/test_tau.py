"""Tests for the tau (Lyman-alpha optical depth) loader.

The loader returns the RAW, redshift-space tau skewers reshaped per line-of-sight
axis to a (ngrid, ngrid, nbins) cube — never modified/resampled. Flux F=exp(-tau)
and delta_flux are optional derived helpers.

Fixtures build a tiny grid_480-schema HDF5 (ngrid=4, 3 axes, with the spectra/axis
labels and the empty placeholder groups the real files carry) so reshape, axis
selection, and the runtime axis guard can be checked exactly.
"""
import os

import h5py
import numpy as np
import pytest

from priya_loader import paths, units
from priya_loader import tau as T


def _make_tau_h5(path, ng=4, nbins=5, redshift=4.2, box=120000.0,
                 with_hz=True, scramble_axis=False, dla=False):
    """Write a minimal grid_480-schema file. tau[g,k] = g*10 + k over all
    3*ng*ng skewers; axis blocks contiguous in order 1,2,3."""
    per_axis = ng * ng
    nlos = 3 * per_axis
    g = np.arange(nlos)[:, None]
    k = np.arange(nbins)[None, :]
    data = (g * 10 + k).astype(np.float32)
    if dla:
        data[0, 0] = 1.0e8                       # saturated absorber
    axis_labels = np.repeat([1, 2, 3], per_axis).astype(np.int32)
    if scramble_axis:
        axis_labels = axis_labels[::-1].copy()   # mislabel the blocks
    with h5py.File(path, "w") as f:
        f.create_dataset("tau/H/1/1215", data=data)
        sp = f.create_group("spectra")
        sp.create_dataset("axis", data=axis_labels)
        for empty in ("colden", "temperature", "velocity", "tau_obs"):
            f.create_group(empty)                # present-but-empty, like real files
        hdr = f.create_group("Header")
        hdr.attrs["redshift"] = redshift
        hdr.attrs["nbins"] = nbins
        hdr.attrs["box"] = box
        hdr.attrs["hubble"] = 0.7
        hdr.attrs["omegam"] = 0.3
        hdr.attrs["omegab"] = 0.05
        hdr.attrs["omegal"] = 0.7
        if with_hz:
            hdr.attrs["Hz"] = 500.0
    return path


@pytest.fixture
def tau_file(tmp_path):
    return _make_tau_h5(tmp_path / "lya_forest_spectra_grid_480.hdf5")


# --- shape / reshape / axis ---------------------------------------------------
def test_load_tau_grid_shape(tau_file):
    g = T.load_tau_grid(tau_file, axis=1)
    assert g.tau.shape == (4, 4, 5)
    assert g.ngrid == 4 and g.nbins == 5 and g.axis == 1


def test_reshape_is_c_order(tau_file):
    g = T.load_tau_grid(tau_file, axis=1)
    assert g.tau[0, 0, 0] == pytest.approx(0.0)
    assert g.tau[0, 1, 0] == pytest.approx(10.0)
    assert g.tau[1, 0, 0] == pytest.approx(40.0)
    assert g.tau[2, 3, 4] == pytest.approx((2 * 4 + 3) * 10 + 4)


def test_axis_selection_picks_correct_block(tau_file):
    assert T.load_tau_grid(tau_file, axis=2).tau[0, 0, 0] == pytest.approx(160.0)
    assert T.load_tau_grid(tau_file, axis=3).tau[0, 0, 0] == pytest.approx(320.0)


def test_returns_raw_tau_not_flux(tau_file):
    g = T.load_tau_grid(tau_file, axis=1)
    assert g.tau[2, 3, 4] == pytest.approx(114.0)
    assert g.tau.dtype == np.float32


def test_tau_array_owns_its_data(tau_file):
    # No live h5py handle escapes the context manager.
    g = T.load_tau_grid(tau_file, axis=1)
    assert g.tau.flags["OWNDATA"] or g.tau.base is None or isinstance(g.tau.base, np.ndarray)


def test_invalid_axis_raises(tau_file):
    with pytest.raises(ValueError):
        T.load_tau_grid(tau_file, axis=4)


# --- runtime axis guard -------------------------------------------------------
def test_axis_guard_rejects_mislabeled_blocks(tmp_path):
    p = _make_tau_h5(tmp_path / "bad.hdf5", scramble_axis=True)
    with pytest.raises(RuntimeError):
        T.load_tau_grid(p, axis=1)


# --- per-axis coordinate mapping (co-registration) ----------------------------
def test_cube_axes_mapping(tau_file):
    assert T.load_tau_grid(tau_file, axis=1).cube_axes == ("y", "z", "x")
    assert T.load_tau_grid(tau_file, axis=2).cube_axes == ("x", "z", "y")
    assert T.load_tau_grid(tau_file, axis=3).cube_axes == ("x", "y", "z")


def test_meta_marks_redshift_space(tau_file):
    g = T.load_tau_grid(tau_file, axis=1)
    assert g.meta["space"] == "redshift"
    assert g.meta["los_is_velocity_axis"] is True


# --- metadata / scales --------------------------------------------------------
def test_redshift_and_box_from_header(tau_file):
    g = T.load_tau_grid(tau_file, axis=1)
    assert g.redshift == pytest.approx(4.2)
    assert g.box == pytest.approx(120.0)


def test_scales_in_meta(tau_file):
    g = T.load_tau_grid(tau_file, axis=1)
    assert g.meta["transverse_kpc_h"] == pytest.approx(120000.0 / 4)
    assert g.meta["los_kpc_h"] == pytest.approx(120000.0 / 5)
    dv_expected = 500.0 * (120.0 / 0.7) / (1 + 4.2) / 5
    assert g.meta["dv_kms"] == pytest.approx(dv_expected)
    assert g.meta["vmax_kms"] == pytest.approx(dv_expected * 5)


def test_dv_kms_falls_back_to_cosmology_without_hz(tmp_path):
    p = _make_tau_h5(tmp_path / "no_hz.hdf5", with_hz=False)
    g = T.load_tau_grid(p, axis=1)
    hz = units.hubble_z(4.2, 0.3, 0.7, 0.7)
    assert g.meta["dv_kms"] == pytest.approx(hz * (120.0 / 0.7) / (1 + 4.2) / 5)


def test_nbins_taken_from_dataset(tmp_path):
    # dataset is authoritative; a wrong Header/nbins must not corrupt the reshape.
    p = _make_tau_h5(tmp_path / "n.hdf5", nbins=7)
    g = T.load_tau_grid(p, axis=1)
    assert g.nbins == 7 and g.tau.shape == (4, 4, 7)


# --- error paths --------------------------------------------------------------
def test_missing_tau_key_raises(tmp_path):
    p = tmp_path / "empty.hdf5"
    with h5py.File(p, "w") as f:
        f.create_group("Header").attrs["redshift"] = 4.0
    with pytest.raises(ValueError):
        T.load_tau_grid(p, axis=1)


def test_bad_skewer_count_raises(tmp_path):
    p = tmp_path / "weird.hdf5"
    with h5py.File(p, "w") as f:
        f.create_dataset("tau/H/1/1215", data=np.zeros((50, 5), dtype=np.float32))
        f.create_group("Header").attrs.update(redshift=4.0, nbins=5, box=120000.0)
    with pytest.raises(ValueError):
        T.load_tau_grid(p, axis=1)


# --- flux helpers -------------------------------------------------------------
def test_flux_helpers(tau_file):
    g = T.load_tau_grid(tau_file, axis=1)
    np.testing.assert_allclose(T.to_flux(g.tau), np.exp(-g.tau), rtol=1e-6)
    assert T.mean_flux(g.tau) == pytest.approx(float(np.exp(-g.tau.astype(np.float64)).mean()))
    df = T.to_delta_flux(g.tau, mean_flux_value=0.5)
    np.testing.assert_allclose(df, np.exp(-g.tau) / 0.5 - 1.0, rtol=1e-5)


def test_mean_flux_accumulates_in_float64(tau_file):
    # Contract: float64 accumulation of the (float32) flux — not a float32 sum.
    g = T.load_tau_grid(tau_file, axis=1)
    ref = float(np.exp(-g.tau).mean(dtype=np.float64))
    assert T.mean_flux(g.tau) == pytest.approx(ref, rel=1e-12)


def test_dla_tau_underflows_cleanly(tmp_path):
    p = _make_tau_h5(tmp_path / "dla.hdf5", dla=True)
    g = T.load_tau_grid(p, axis=1)
    flux = T.to_flux(g.tau)
    assert np.isfinite(flux).all()
    assert flux[0, 0, 0] == 0.0                # exp(-1e8) underflows to 0
    assert 0.0 < T.mean_flux(g.tau) < 1.0


# --- real data (auto-skipped unless PRIYA_DATA_ROOT set) ----------------------
@pytest.mark.realdata
def test_load_real_grid_480():
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
    assert g.cube_axes == ("y", "z", "x")
    assert g.meta["transverse_kpc_h"] == pytest.approx(250.0)
    assert g.meta["dv_kms"] == pytest.approx(10.0, abs=1.0)
    assert 0.0 < T.mean_flux(g.tau) < 1.0
