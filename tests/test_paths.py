"""Tests for discovering simulations and locating IC / flux files on disk.

All tests use synthetic directory trees (``tmp_path``) so they run anywhere,
including a NERSC login node with no real data staged. Robustness to partial /
mid-transfer data (empty SPECTRA dirs, missing IC grids) is explicitly covered.
"""
import json

import pytest

from priya_loader import paths

NAME_A = (
    "ns0.905Ap1.79e-09herei3.97heref2.99alphaq2.1hub0.745"
    "omegamh20.145hireionz7.47bhfeedback0.043"
)
NAME_B = (
    "ns0.842Ap1.36e-09herei3.51heref2.85alphaq2hub0.658"
    "omegamh20.14hireionz6.72bhfeedback0.0453"
)


def _make_sim(root, name):
    d = root / name
    (d / "output").mkdir(parents=True)
    (d / "SimulationICs.json").write_text(json.dumps({"box": 120, "npart": 1536}))
    return d


def test_discover_simulations_finds_only_sim_dirs(tmp_path):
    _make_sim(tmp_path, NAME_A)
    _make_sim(tmp_path, NAME_B)
    # decoys that must be ignored
    (tmp_path / "emulator_params.json").write_text("{}")
    (tmp_path / "MP-GenIC_bio").mkdir()
    sims = paths.discover_simulations(tmp_path)
    assert [s.name for s in sims] == sorted([NAME_A, NAME_B])
    assert all((s.directory / "SimulationICs.json").exists() for s in sims)


def test_discover_simulations_empty_root(tmp_path):
    assert paths.discover_simulations(tmp_path) == []


def test_find_ic_dir_prefers_largest_grid(tmp_path):
    d = _make_sim(tmp_path, NAME_A)
    for g in ("120_512_99", "120_1536_99"):
        (d / "ICS" / g).mkdir(parents=True)
    got = paths.find_ic_dir(d)
    assert got.name == "120_1536_99"
    # explicit request honoured
    assert paths.find_ic_dir(d, ngrid=512).name == "120_512_99"


def test_find_ic_dir_fallback_when_only_companion_present(tmp_path):
    d = _make_sim(tmp_path, NAME_A)
    (d / "ICS" / "120_512_99").mkdir(parents=True)
    assert paths.find_ic_dir(d).name == "120_512_99"
    assert paths.find_ic_dir(d, ngrid=1536) is None


def test_find_ic_dir_absent(tmp_path):
    d = _make_sim(tmp_path, NAME_A)
    assert paths.find_ic_dir(d) is None


def _add_spectra(sim_dir, snap, *, with_grid=True, empty=False):
    sd = sim_dir / "output" / f"SPECTRA_{snap:03d}"
    sd.mkdir(parents=True)
    if with_grid:
        f = sd / paths.GRID_SPECTRA_FILENAME
        f.write_bytes(b"" if empty else b"\x89HDF\r\n\x1a\n")
    return sd


def test_find_spectra_files_sorted_and_indexed(tmp_path):
    d = _make_sim(tmp_path, NAME_A)
    _add_spectra(d, 6)
    _add_spectra(d, 4)
    found = paths.find_spectra_files(d)
    assert [snap for snap, _ in found] == [4, 6]            # sorted by index
    assert all(p.name == paths.GRID_SPECTRA_FILENAME for _, p in found)


def test_find_spectra_files_skips_empty_and_missing(tmp_path):
    d = _make_sim(tmp_path, NAME_A)
    _add_spectra(d, 4)                       # good
    _add_spectra(d, 5, empty=True)           # 0-byte grid file -> skip
    _add_spectra(d, 6, with_grid=False)      # SPECTRA dir but no grid file -> skip
    found = paths.find_spectra_files(d)
    assert [snap for snap, _ in found] == [4]


def test_find_spectra_files_none(tmp_path):
    d = _make_sim(tmp_path, NAME_A)
    assert paths.find_spectra_files(d) == []


def test_find_spectra_files_handles_index_gaps(tmp_path):
    # Real data has gaps (e.g. SPECTRA_019 missing); return present indices only.
    d = _make_sim(tmp_path, NAME_A)
    for snap in (4, 5, 6, 17, 18, 20, 21, 22):   # 19 deliberately absent
        _add_spectra(d, snap)
    assert [s for s, _ in paths.find_spectra_files(d)] == [4, 5, 6, 17, 18, 20, 21, 22]


def _add_runfiles(sim_dir, ngrid=1536, box_kpc=120000):
    (sim_dir / "mpgadget.param").write_text(f"InitCondFile = ICS/{box_kpc//1000}_{ngrid}_99\n")
    (sim_dir / "_genic_params.ini").write_text(f"Ngrid = {ngrid}\nBoxSize = {box_kpc}\nNgridNu = 0\n")


def test_find_production_ic_dir_none_when_only_companion(tmp_path):
    d = _make_sim(tmp_path, NAME_A)
    _add_runfiles(d, ngrid=1536)
    (d / "ICS" / "120_512_99").mkdir(parents=True)        # only the companion staged
    assert paths.find_production_ic_dir(d) is None


def test_find_production_ic_dir_returns_staged_production(tmp_path):
    d = _make_sim(tmp_path, NAME_A)
    _add_runfiles(d, ngrid=1536)
    (d / "ICS" / "120_1536_99").mkdir(parents=True)
    got = paths.find_production_ic_dir(d)
    assert got is not None and got.name == "120_1536_99"
