"""Tests for parsing simulation parameters from run configs + ICs JSON.

Box/Ngrid come from the authoritative run files (mpgadget.param /
_genic_params.ini), NOT from the often-stale SimulationICs.json. Cosmology /
astro params come from the JSON.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from priya_loader import params as P

LOWRES_NAME = (
    "ns0.905Ap1.79e-09herei3.97heref2.99alphaq2.1hub0.745"
    "omegamh20.145hireionz7.47bhfeedback0.043"
)
LOWRES_JSON = {
    "box": 120, "npart": 1536,
    "omega0": 0.26206025134002975, "omegab": 0.04035854240799964,
    "hubble": 0.745, "scalar_amp": 2.3227783434509894e-09, "ns": 0.904734358,
    "redshift": 99, "redend": 2.0,
    "here_i": 3.97459518, "here_f": 2.98720962, "alpha_q": 2.09592901,
    "hireionz": 7.46623344, "bhfeedback": 0.0430153024, "heatamp": 1.0,
}


def _write_sim(tmp_path, name, jsondict, *, ngrid=1536, box_kpc=120000, runfiles=True):
    d = tmp_path / name
    d.mkdir()
    (d / "SimulationICs.json").write_text(json.dumps(jsondict))
    if runfiles:
        (d / "mpgadget.param").write_text(
            f"InitCondFile = ICS/{box_kpc // 1000}_{ngrid}_99\nOmega0 = {jsondict.get('omega0', 0.26)}\n"
        )
        (d / "_genic_params.ini").write_text(
            f"Ngrid = {ngrid}\nBoxSize = {box_kpc}\nNgridNu = 0\n"
        )
    return d


# --- name parsing -------------------------------------------------------------
def test_parse_sim_name_extracts_nine_params():
    p = P.parse_sim_name(LOWRES_NAME)
    assert p["ns"] == pytest.approx(0.905)
    assert p["herei"] == pytest.approx(3.97)
    assert p["alphaq"] == pytest.approx(2.1)
    assert p["hub"] == pytest.approx(0.745)
    assert p["omegamh2"] == pytest.approx(0.145)
    assert p["bhfeedback"] == pytest.approx(0.043)


def test_parse_sim_name_handles_scientific_amplitude():
    assert P.parse_sim_name(LOWRES_NAME)["Ap"] == pytest.approx(1.79e-09)


def test_parse_sim_name_rejects_garbage():
    with pytest.raises(ValueError):
        P.parse_sim_name("not_a_sim_folder")


def test_parse_sim_name_rejects_trailing_junk():
    with pytest.raises(ValueError):
        P.parse_sim_name(LOWRES_NAME + "_v2_backup")


# --- the BLOCKER regression ---------------------------------------------------
def test_box_npart_come_from_runfiles_not_stale_json(tmp_path):
    stale = dict(LOWRES_JSON, box=15, npart=192)   # the real stale-template values
    d = _write_sim(tmp_path, LOWRES_NAME, stale, ngrid=1536, box_kpc=120000)
    sp = P.SimParams.from_dir(d)
    assert sp.box == pytest.approx(120.0)          # NOT 15
    assert sp.npart == 1536                         # NOT 192
    assert sp.fidelity == "lowres"


# --- canonical fields ---------------------------------------------------------
def test_simparams_from_dir_reads_cosmology_from_json(tmp_path):
    d = _write_sim(tmp_path, LOWRES_NAME, LOWRES_JSON)
    sp = P.SimParams.from_dir(d)
    assert sp.name == LOWRES_NAME
    assert sp.box == pytest.approx(120)
    assert sp.npart == 1536
    assert sp.hubble == pytest.approx(0.745)
    assert sp.ns == pytest.approx(0.904734358)
    assert sp.z_init == pytest.approx(99)
    assert sp.z_end == pytest.approx(2.0)
    assert sp.here_i == pytest.approx(3.97459518)
    assert sp.bhfeedback == pytest.approx(0.0430153024)


def test_fidelity_from_production_ngrid(tmp_path):
    d_lo = _write_sim(tmp_path, LOWRES_NAME, LOWRES_JSON, ngrid=1536)
    assert P.SimParams.from_dir(d_lo).fidelity == "lowres"
    d_hi = _write_sim(tmp_path, "hi_" + LOWRES_NAME, LOWRES_JSON, ngrid=3072)
    assert P.SimParams.from_dir(d_hi).fidelity == "hires"


def test_omega_lambda_and_omega_m_h2(tmp_path):
    d = _write_sim(tmp_path, LOWRES_NAME, LOWRES_JSON)
    sp = P.SimParams.from_dir(d)
    assert sp.omega_lambda == pytest.approx(1.0 - sp.omega0)   # flat LCDM
    assert sp.omega_m_h2 == pytest.approx(sp.omega0 * sp.hubble**2)
    assert sp.omega_m_h2 == pytest.approx(0.145, rel=0.01)


# --- validation ---------------------------------------------------------------
def test_resolution_invariant_rejects_unknown_grid(tmp_path):
    d = _write_sim(tmp_path, LOWRES_NAME, LOWRES_JSON, ngrid=999)
    with pytest.raises(ValueError):
        P.SimParams.from_dir(d)


def test_resolution_invariant_rejects_wrong_box(tmp_path):
    d = _write_sim(tmp_path, LOWRES_NAME, LOWRES_JSON, ngrid=1536, box_kpc=60000)
    with pytest.raises(ValueError):
        P.SimParams.from_dir(d)


def test_amplitude_pivot_check_passes_for_consistent_sim(tmp_path):
    d = _write_sim(tmp_path, LOWRES_NAME, LOWRES_JSON)
    P.SimParams.from_dir(d)  # must not raise


def test_amplitude_pivot_check_rejects_mismatch(tmp_path):
    bad = dict(LOWRES_JSON, scalar_amp=5.0e-09)  # folder Ap says ~1.79e-9
    d = _write_sim(tmp_path, LOWRES_NAME, bad)
    with pytest.raises(ValueError):
        P.SimParams.from_dir(d)


def test_gross_cosmology_mismatch_raises(tmp_path):
    bad = dict(LOWRES_JSON, hubble=0.5)  # name says hub0.745
    d = _write_sim(tmp_path, LOWRES_NAME, bad)
    with pytest.raises(ValueError):
        P.SimParams.from_dir(d)


# --- ergonomics ---------------------------------------------------------------
def test_simparams_is_hashable(tmp_path):
    d = _write_sim(tmp_path, LOWRES_NAME, LOWRES_JSON)
    sp = P.SimParams.from_dir(d)
    assert hash(sp) == hash(sp)
    assert len({sp, sp}) == 1            # usable as set/dict key


def test_directory_is_path(tmp_path):
    d = _write_sim(tmp_path, LOWRES_NAME, LOWRES_JSON)
    assert isinstance(P.SimParams.from_dir(d).directory, Path)


def test_missing_json_raises_valueerror(tmp_path):
    d = tmp_path / LOWRES_NAME
    d.mkdir()
    (d / "mpgadget.param").write_text("InitCondFile = ICS/120_1536_99\n")
    with pytest.raises(ValueError):
        P.SimParams.from_dir(d)


def test_missing_required_cosmo_key_raises(tmp_path):
    incomplete = {k: v for k, v in LOWRES_JSON.items() if k != "omega0"}
    d = _write_sim(tmp_path, LOWRES_NAME, incomplete)
    with pytest.raises(ValueError):
        P.SimParams.from_dir(d)


def test_as_vector_orders_by_keys(tmp_path):
    d = _write_sim(tmp_path, LOWRES_NAME, LOWRES_JSON)
    sp = P.SimParams.from_dir(d)
    v = sp.as_vector(["ns", "hubble", "bhfeedback"])
    np.testing.assert_allclose(v, [0.904734358, 0.745, 0.0430153024])
