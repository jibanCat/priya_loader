"""Tests for the PriyaDataset orchestrator.

Builds a tiny but complete synthetic simulation tree (folder name + JSON +
run-config files + a small IC bigfile + small tau grid_480 files) and checks the
dataset loops over (sim, redshift), pairs params + IC + tau, degrades gracefully
on partial data, and exports per-(sim, z) .npz. Requires bigfile + h5py.
"""
import json

import numpy as np
import pytest

bigfile = pytest.importorskip("bigfile")
import h5py  # noqa: E402

from priya_loader import dataset as ds

NAME_A = (
    "ns0.905Ap1.79e-09herei3.97heref2.99alphaq2.1hub0.745"
    "omegamh20.145hireionz7.47bhfeedback0.043"
)
NAME_B = (
    "ns0.842Ap1.36e-09herei3.51heref2.85alphaq2hub0.658"
    "omegamh20.14hireionz6.72bhfeedback0.0453"
)
JSON_A = {
    "box": 15, "npart": 192,                      # stale on purpose (run files win)
    "omega0": 0.26206025134002975, "omegab": 0.04035854240799964,
    "hubble": 0.745, "scalar_amp": 2.3227783434509894e-09, "ns": 0.904734358,
    "redshift": 99, "redend": 2.0, "here_i": 3.97459518, "here_f": 2.98720962,
    "alpha_q": 2.09592901, "hireionz": 7.46623344, "bhfeedback": 0.0430153024,
}


def _write_tau(path, ng=4, nbins=5, redshift=4.2, box=120000.0):
    per = ng * ng
    nlos = 3 * per
    g = np.arange(nlos)[:, None]
    k = np.arange(nbins)[None, :]
    with h5py.File(path, "w") as f:
        f.create_dataset("tau/H/1/1215", data=(g * 10 + k).astype(np.float32))
        f.create_group("spectra").create_dataset(
            "axis", data=np.repeat([1, 2, 3], per).astype(np.int32))
        h = f.create_group("Header")
        h.attrs.update(redshift=redshift, nbins=nbins, box=box,
                       hubble=0.745, omegam=0.26, omegab=0.04, omegal=0.74, Hz=500.0)


def _write_ic(path, ng=8, box=120000.0):
    g = (np.arange(ng) + 0.5) * box / ng
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype("f8")
    n = pos.shape[0]
    ii = np.arange(n) // (ng * ng)
    with bigfile.File(str(path), create=True) as bf:
        bf.create_from_array("1/Position", pos)
        bf.create_from_array("1/ID", (np.arange(n) + 1).astype("u8"))
        bf.create_from_array("1/ICDensity", np.sin(2 * np.pi * ii / ng).astype("f4"))
        bf.create("Header")
        bf["Header"].attrs["BoxSize"] = box
        bf["Header"].attrs["Redshift"] = 99.0
        bf["Header"].attrs["HubbleParam"] = 0.745
        bf["Header"].attrs["Omega0"] = 0.262
        bf["Header"].attrs["OmegaLambda"] = 0.738


def _make_sim(root, name, jsondict, *, snaps=(4,), with_ic=True, ngrid=1536,
              broken_ic=False, bad_tau=False):
    d = root / name
    (d / "output").mkdir(parents=True)
    (d / "SimulationICs.json").write_text(json.dumps(jsondict))
    (d / "mpgadget.param").write_text(f"InitCondFile = ICS/120_{ngrid}_99\n")
    (d / "_genic_params.ini").write_text(f"Ngrid = {ngrid}\nBoxSize = 120000\nNgridNu = 0\n")
    if broken_ic:
        (d / "ICS" / f"120_{ngrid}_99").mkdir(parents=True)   # skeleton dir, no data
    elif with_ic:
        _write_ic(d / "ICS" / f"120_{ngrid}_99")
    for i, s in enumerate(snaps):
        sd = d / "output" / f"SPECTRA_{s:03d}"
        sd.mkdir(parents=True)
        f = sd / "lya_forest_spectra_grid_480.hdf5"
        if bad_tau:
            f.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)   # truncated/corrupt
        else:
            _write_tau(f, redshift=5.0 - 0.2 * i)
    return d


def test_iter_yields_params_z_ic_tau(tmp_path):
    _make_sim(tmp_path, NAME_A, JSON_A, snaps=(4,))
    samples = list(ds.PriyaDataset(tmp_path, ic_nmesh=8).iter_samples())
    assert len(samples) == 1
    s = samples[0]
    assert s.params.name == NAME_A
    assert s.params.npart == 1536          # from run files, not the stale JSON
    assert s.redshift == pytest.approx(5.0)
    assert s.tau.shape == (4, 4, 5)
    assert s.ic.shape == (8, 8, 8)


def test_multiple_redshifts_per_sim(tmp_path):
    _make_sim(tmp_path, NAME_A, JSON_A, snaps=(4, 5, 6))
    samples = list(ds.PriyaDataset(tmp_path, ic_nmesh=8))
    assert len(samples) == 3
    assert {round(s.redshift, 1) for s in samples} == {5.0, 4.8, 4.6}


def test_ic_loaded_once_shared_across_redshifts(tmp_path):
    _make_sim(tmp_path, NAME_A, JSON_A, snaps=(4, 5))
    samples = list(ds.PriyaDataset(tmp_path, ic_nmesh=8))
    assert samples[0].ic is samples[1].ic     # same array object reused per sim


def test_ic_none_when_absent(tmp_path):
    _make_sim(tmp_path, NAME_A, JSON_A, snaps=(4,), with_ic=False)
    s = list(ds.PriyaDataset(tmp_path, ic_nmesh=8))[0]
    assert s.ic is None
    assert s.tau is not None


def test_load_ic_false_skips_ic(tmp_path):
    _make_sim(tmp_path, NAME_A, JSON_A, snaps=(4,))
    s = list(ds.PriyaDataset(tmp_path, load_ic=False))[0]
    assert s.ic is None


def test_skips_sim_without_tau(tmp_path):
    _make_sim(tmp_path, NAME_A, JSON_A, snaps=())
    assert list(ds.PriyaDataset(tmp_path, ic_nmesh=8)) == []


def test_discovers_multiple_sims_sorted(tmp_path):
    # Two discoverable sims; validate=False since NAME_B's tokens != JSON_A cosmology
    # (this test exercises discovery/sorting, not the param consistency check).
    _make_sim(tmp_path, NAME_A, JSON_A, snaps=(4,))
    _make_sim(tmp_path, NAME_B, JSON_A, snaps=(4,))
    names = [s.params.name for s in ds.PriyaDataset(tmp_path, ic_nmesh=8, validate=False)]
    assert names == sorted([NAME_A, NAME_B])


def test_invalid_sim_is_skipped_with_warning(tmp_path):
    # A folder whose name disagrees with its JSON cosmology is skipped (not crash).
    _make_sim(tmp_path, NAME_A, JSON_A, snaps=(4,))            # valid
    _make_sim(tmp_path, NAME_B, JSON_A, snaps=(4,))            # NAME_B tokens != JSON_A
    with pytest.warns(Warning):
        names = [s.params.name for s in ds.PriyaDataset(tmp_path, ic_nmesh=8)]
    assert names == [NAME_A]                                   # NAME_B skipped


def test_export_npz_roundtrip(tmp_path):
    _make_sim(tmp_path, NAME_A, JSON_A, snaps=(4,))
    out = tmp_path / "out"
    paths = ds.PriyaDataset(tmp_path, ic_nmesh=8).export(out)
    assert len(paths) == 1
    z = np.load(paths[0], allow_pickle=True)
    assert z["tau"].shape == (4, 4, 5)
    assert z["ic"].shape == (8, 8, 8)
    assert float(z["redshift"]) == pytest.approx(5.0)
    params = json.loads(str(z["params"]))
    # resolved params (authoritative), not the stale JSON box=15/npart=192
    assert params["box"] == pytest.approx(120.0)
    assert params["npart"] == 1536
    meta = json.loads(str(z["meta"]))
    assert tuple(meta["tau_cube_axes"]) == ("y", "z", "x")   # co-registration info retained


def test_ic_field_icdensity_through_dataset(tmp_path):
    # The headline orchestrator can deliver Roger's linear delta_1.
    _make_sim(tmp_path, NAME_A, JSON_A, snaps=(4,))
    s = list(ds.PriyaDataset(tmp_path, ic_nmesh=8, ic_field="icdensity"))[0]
    assert s.meta["ic_meta"]["field"] == "icdensity"
    expected = np.sin(2 * np.pi * np.arange(8) / 8).astype(np.float32)
    np.testing.assert_allclose(s.ic[:, 0, 0], expected, atol=1e-6)


def test_graceful_on_broken_ic(tmp_path):
    # A skeleton (data-less) production IC dir must yield ic=None, not crash.
    _make_sim(tmp_path, NAME_A, JSON_A, snaps=(4,), broken_ic=True)
    with pytest.warns(Warning):
        s = list(ds.PriyaDataset(tmp_path, ic_nmesh=8))[0]
    assert s.ic is None
    assert s.tau is not None


def test_graceful_on_truncated_tau(tmp_path):
    # A truncated tau file is skipped (warn), not fatal.
    _make_sim(tmp_path, NAME_A, JSON_A, snaps=(4, 5), bad_tau=True)
    with pytest.warns(Warning):
        samples = list(ds.PriyaDataset(tmp_path, ic_nmesh=8))
    assert samples == []          # both snapshots truncated -> skipped, no crash


def test_shared_ic_is_read_only(tmp_path):
    _make_sim(tmp_path, NAME_A, JSON_A, snaps=(4,))
    s = list(ds.PriyaDataset(tmp_path, ic_nmesh=8))[0]
    assert s.ic.flags.writeable is False   # protects the IC shared across redshifts
