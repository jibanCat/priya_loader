"""Tests for reading the *authoritative* run configuration (box, Ngrid).

Critical: the per-simulation ``SimulationICs.json`` carries STALE template
``box``/``npart`` for ~half the suite (box=15, npart=192). The truth lives in
``mpgadget.param`` (``InitCondFile = ICS/<box>_<Ngrid>_99``) and
``_genic_params.ini`` (``BoxSize``, ``Ngrid``). These tests pin that behaviour.
"""
import pytest

from priya_loader import runconfig as rc

MPGADGET = """\
InitCondFile = ICS/120_1536_99
OutputDir = output
Omega0 = 0.26206025134002975
HubbleParam = 0.745
"""
GENIC = """\
OutputDir = ICS
FileBase = 120_1536_99
Ngrid = 1536
BoxSize = 120000
NgridNu = 0
Omega0 = 0.26206025134002975
"""


def _write(tmp_path, mpgadget=MPGADGET, genic=GENIC):
    if mpgadget is not None:
        (tmp_path / "mpgadget.param").write_text(mpgadget)
    if genic is not None:
        (tmp_path / "_genic_params.ini").write_text(genic)
    return tmp_path


def test_reads_production_box_and_ngrid(tmp_path):
    _write(tmp_path)
    cfg = rc.read_run_config(tmp_path)
    assert cfg.box == pytest.approx(120.0)     # Mpc/h, from BoxSize=120000 kpc/h
    assert cfg.ngrid == 1536
    assert cfg.ic_basename == "120_1536_99"


def test_records_box_and_ngrid_provenance(tmp_path):
    # box comes from genic BoxSize (exact float); ngrid from mpgadget InitCondFile.
    _write(tmp_path)
    cfg = rc.read_run_config(tmp_path)
    assert cfg.box_source == "_genic_params.ini"
    assert cfg.ngrid_source == "mpgadget.param"


def test_provenance_when_only_genic(tmp_path):
    _write(tmp_path, mpgadget=None)
    cfg = rc.read_run_config(tmp_path)
    assert cfg.box_source == "_genic_params.ini"
    assert cfg.ngrid_source == "_genic_params.ini"


def test_ngrid_not_confused_by_ngridnu(tmp_path):
    # A naive 'startswith("Ngrid")' would read NgridNu=0; must get 1536.
    _write(tmp_path)
    assert rc.read_run_config(tmp_path).ngrid == 1536


def test_ignores_stale_json_values(tmp_path):
    # Even with a stale JSON present, run config is taken from the run files.
    (tmp_path / "SimulationICs.json").write_text('{"box": 15, "npart": 192}')
    _write(tmp_path)
    cfg = rc.read_run_config(tmp_path)
    assert cfg.box == pytest.approx(120.0)
    assert cfg.ngrid == 1536


def test_hires_config(tmp_path):
    _write(
        tmp_path,
        mpgadget="InitCondFile = ICS/120_3072_99\n",
        genic="Ngrid = 3072\nBoxSize = 120000\nNgridNu = 0\n",
    )
    cfg = rc.read_run_config(tmp_path)
    assert cfg.ngrid == 3072
    assert cfg.box == pytest.approx(120.0)


def test_mismatch_between_mpgadget_and_genic_raises(tmp_path):
    _write(tmp_path, mpgadget="InitCondFile = ICS/120_1536_99\n",
           genic="Ngrid = 512\nBoxSize = 120000\nNgridNu = 0\n")
    with pytest.raises(ValueError):
        rc.read_run_config(tmp_path)


def test_falls_back_to_genic_when_mpgadget_absent(tmp_path):
    _write(tmp_path, mpgadget=None)
    cfg = rc.read_run_config(tmp_path)
    assert cfg.ngrid == 1536
    assert cfg.box == pytest.approx(120.0)


def test_missing_both_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        rc.read_run_config(tmp_path)
