"""Real-IC verification — the sign-off checks that can only run where MP-GenIC
IC bigfiles are actually **staged** (not the ~26 KB empty skeletons present on the
Great Lakes test scratch).

Everything here is gated behind the ``PRIYA_REAL_IC`` environment variable and
**skips** when it is unset, so the default suite stays green. The IC path has so
far been verified only on *synthetic* bigfiles plus an explicit/Pylians CIC
reference on *synthetic* random points; these tests close that gap by exercising
the loader and the CIC painter on **real** particle data (real comoving-kpc/h
coordinates, real periodic box, real Lagrangian ``ID``/``ICDensity`` blocks).

Run on NERSC, e.g.::

    PRIYA_REAL_IC=/global/cfs/cdirs/.../<sim>/ICS/120_512_99 \\
        pytest tests/test_real_ic.py -v

The staged PRIYA low-res IC is the **512^3** companion ``120_512_99`` (the 1536^3
production grid is not staged on NERSC); the hi-res IC is ``120_3072_99`` (+ a
``120_1024_99`` companion). For a hi-res target run the resolution/box test only —
it reads no particles; the full-block fixtures self-skip above 1536^3 (override
with ``PRIYA_REAL_IC_ALLOW_HEAVY=1`` on a compute node).

The built-in-CIC-vs-explicit-reference check runs with **no optional dependency**,
so the convention/origin/transpose guarantee is verified on the default stack and
is never silently skipped. The Pylians/nbodykit checks are *extra* cross-checks and
additionally need those libraries (Pylians builds only in a ``numpy<2`` env).
Optional ``PRIYA_REAL_IC_SUBSAMPLE`` (default 4000) bounds the particle load; the
CIC painter is additive per particle, so equivalence on a strided subsample implies
equivalence on the full set (and keeps RAM bounded).

Heavy loads (one full IC stream per mesh) are shared via module-scoped fixtures.

All tests here use ``ptype="dm"`` (the total-matter proxy). The gas (type-0) route
is code-symmetric (``PTYPE["gas"]=0``) and is covered on synthetic fixtures; it is
not separately asserted against a real IC.
"""
import os

import numpy as np
import pytest

from priya_loader import load_ic_density, load_ic_particles, mesh
from test_mesh import _explicit_cic   # reuse the transparent, dependency-free CIC reference

REAL_IC = os.environ.get("PRIYA_REAL_IC")
SUBSAMPLE = int(os.environ.get("PRIYA_REAL_IC_SUBSAMPLE", "4000"))
BOX_MPC_H = 120.0
BOX_KPC_H = BOX_MPC_H * 1000.0
ALLOWED_NGRID = {512, 1024, 1536, 3072}   # 512/1024 companions + low-res / hi-res production
NMESH = 256                               # divides 512, 1024, 1536 and 3072

pytestmark = pytest.mark.skipif(
    not REAL_IC,
    reason="set PRIYA_REAL_IC to a staged ICS/<box>_<Ngrid>_99 directory to run",
)


def _cube_root(n):
    """Exact integer cube root, robust to float rounding at 3072^3 ~ 2.9e10."""
    g = int(round(n ** (1.0 / 3.0)))
    for c in (g - 1, g, g + 1):
        if c > 0 and c ** 3 == n:
            return c
    raise AssertionError(f"{n} particles is not a perfect cube (Ngrid^3)")


def _ngrid_from_header():
    """Ngrid from the IC Header (reads only the block record count, no particles)."""
    bigfile = pytest.importorskip("bigfile")
    with bigfile.File(str(REAL_IC)) as bf:
        return _cube_root(int(bf["1/Position"].size))


def _skip_if_heavy_load():
    """Refuse to stream a full hi-res Position block by accident. The full-block
    fixtures read the ENTIRE Position array (``subsample`` strides only in memory),
    so a 3072^3 target is ~700 GB of I/O — never appropriate on a shared login node.
    Resolution/box tests don't use these fixtures, so they still run at any Ngrid."""
    ng = _ngrid_from_header()
    if ng > 1536 and not os.environ.get("PRIYA_REAL_IC_ALLOW_HEAVY"):
        pytest.skip(
            f"Ngrid={ng}: a full-block load is ~{ng ** 3 * 24 / 1e9:.0f} GB of I/O; "
            "set PRIYA_REAL_IC_ALLOW_HEAVY=1 on a compute node to override"
        )


# --- shared heavy loads -------------------------------------------------------
@pytest.fixture(scope="module")
def real_particles():
    """A memory-bounded, strided subsample of the real DM ``Position`` block
    (comoving kpc/h, float64) plus the box in kpc/h."""
    _skip_if_heavy_load()
    data, header = load_ic_particles(
        REAL_IC, ptype="dm", columns=("Position",), subsample=SUBSAMPLE
    )
    return data["Position"].astype(np.float64), float(header["box_kpc_h"])


@pytest.fixture(scope="module")
def ic_icdensity():
    """The native linear ``delta_1`` field (``field='icdensity'``), loaded once."""
    _skip_if_heavy_load()
    return load_ic_density(REAL_IC, ptype="dm", nmesh=NMESH, field="icdensity")


@pytest.fixture(scope="module")
def ic_cic():
    """The Eulerian CIC overdensity (``field='cic'``), loaded once."""
    _skip_if_heavy_load()
    return load_ic_density(REAL_IC, ptype="dm", nmesh=NMESH, field="cic")


# --- resolution / unit invariants (repo hard constraints) --------------------
def test_real_ic_resolution_and_box_invariants():
    """npart == Ngrid^3 with Ngrid in {512,1024,1536,3072}, and box == 120 Mpc/h.
    Reads only the block size + Header (no painting) — safe at any resolution."""
    bigfile = pytest.importorskip("bigfile")
    with bigfile.File(str(REAL_IC)) as bf:
        npart = int(bf["1/Position"].size)              # bigfile .size = record count
        box_kpc_h = float(np.asarray(bf["Header"].attrs["BoxSize"]).ravel()[0])
    assert _cube_root(npart) in ALLOWED_NGRID, f"unexpected Ngrid for npart={npart}"
    assert np.isclose(box_kpc_h, BOX_KPC_H), f"box {box_kpc_h} kpc/h != {BOX_KPC_H}"


def test_real_ic_positions_within_box_and_units(real_particles):
    """Real ``Position`` is comoving kpc/h, shape (N,3), inside [0, box]."""
    pos, box_kpc_h = real_particles
    assert np.isclose(box_kpc_h, BOX_KPC_H)
    assert pos.ndim == 2 and pos.shape[1] == 3
    assert pos.min() >= 0.0
    assert pos.max() <= box_kpc_h * (1.0 + 1e-6)


# --- the linear delta_1 (icdensity route) -------------------------------------
def test_real_ic_icdensity_is_zero_mean_structured_real_field(ic_icdensity):
    """field='icdensity' -> linear delta_1: finite, ~zero-mean density contrast,
    with real structure (the MP-GenIC realization of the CLASS linear P(k))."""
    d = ic_icdensity.delta
    assert d.shape == (NMESH, NMESH, NMESH)
    assert np.isfinite(d).all()
    assert abs(float(d.mean())) < 0.05            # it's a density contrast
    assert float(d.std()) > 0.0                   # not flat
    assert ic_icdensity.space == "real"
    assert _cube_root(ic_icdensity.npart) in ALLOWED_NGRID


# --- the Eulerian CIC field ---------------------------------------------------
def test_real_ic_cic_overdensity_is_physical(ic_cic):
    """field='cic' -> rho/<rho>-1: mean ~ 0, min >= -1, clusters (max > 0)."""
    d = ic_cic.delta
    assert abs(float(d.mean())) < 1e-3
    assert float(d.min()) >= -1.0 - 1e-6
    assert float(d.max()) > 0.0
    assert float(d.std()) > 0.0


# --- CIC bit-equivalence on REAL particles (the headline verification) --------
def test_real_ic_cic_matches_explicit_reference_on_real_particles(real_particles):
    """Built-in CIC == the transparent per-particle reference (``test_mesh._explicit_cic``)
    on REAL Positions, with **no optional dependency**, so it runs on the default stack
    (it uses the heavy fixture, so it self-skips on 3072^3 without PRIYA_REAL_IC_ALLOW_HEAVY).
    ``_explicit_cic`` shares this painter's exact CIC convention by construction, so it
    catches indexing/flatten/wrap/vectorization bugs but NOT a shared-convention error;
    the Pylians/nbodykit cross-checks below are the convention-independent anchors. A
    bug here would be O(1)."""
    pos, box = real_particles
    grid = 128
    chk = pos[:30000]                                   # bound the O(N) python reference
    ours = mesh.cic_paint(chk, grid, boxsize=box)
    assert np.isclose(ours.sum(), len(chk))             # mass conserved on real coords
    ref = _explicit_cic(chk, grid, boxsize=box)
    np.testing.assert_allclose(ours, ref, atol=1e-9, rtol=0.0)


def test_real_ic_cic_matches_pylians_on_real_particles(real_particles):
    """Extra cross-check: built-in numpy CIC == Pylians (MAS_library) CIC on REAL
    Positions, to float32 accumulation roundoff. Auto-skips without Pylians (which
    builds only in a numpy<2 env); the explicit-reference test above is the
    dependency-free guarantee. (Pylians MA(...,"CIC") convention pinned in PROVENANCE.md.)"""
    MASL = pytest.importorskip("MAS_library")
    pos, box = real_particles
    grid = 128
    rho = mesh.cic_paint(pos, grid, boxsize=box)
    assert np.isclose(rho.sum(), len(pos))             # mass conserved on real coords
    ours = rho / rho.mean() - 1.0
    pyl = np.zeros((grid, grid, grid), dtype=np.float32)
    MASL.MA(pos.astype(np.float32), pyl, box, "CIC")
    theirs = pyl / pyl.mean() - 1.0
    np.testing.assert_allclose(ours, theirs, atol=1e-3, rtol=1e-3)


def test_real_ic_cic_matches_nbodykit_on_real_particles(real_particles):
    """Extra cross-check: built-in CIC == nbodykit's UNcompensated, NON-interlaced
    CIC on REAL Positions (the 'tricks' we deliberately omit). Auto-skips without
    nbodykit. (nbodykit to_mesh(compensated=False, interlaced=False) convention pinned in PROVENANCE.md.)"""
    pytest.importorskip("nbodykit")
    from nbodykit.source.catalog import ArrayCatalog
    pos, box = real_particles
    grid = 128
    ours = mesh.to_overdensity(mesh.cic_paint(pos, grid, boxsize=box))
    cat = ArrayCatalog({"Position": pos})
    m = cat.to_mesh(Nmesh=grid, BoxSize=box, resampler="cic",
                    compensated=False, interlaced=False)
    theirs = np.asarray(m.compute(mode="real")) - 1.0   # nbodykit paints 1+delta
    np.testing.assert_allclose(ours, theirs, atol=1e-3, rtol=1e-3)


# --- consistency of the two IC routes -----------------------------------------
def test_real_ic_icdensity_and_cic_trace_same_structure(ic_icdensity, ic_cic):
    """At z_init~99 the Zel'dovich displacements are sub-cell, so the linear
    (icdensity) and Eulerian (cic) fields trace the same structure -> strongly
    correlated. Guards against a mis-mapped Lagrangian index in the icdensity
    route (which would decorrelate them). 0.5 was only a loose anti-transpose
    guard; at z~99 the correlation is high (measured r≈0.87 at nmesh=256; the CIC
    window pulls it below 1), so we log r and require a floor of 0.8."""
    a = (ic_icdensity.delta - ic_icdensity.delta.mean()).ravel()
    b = (ic_cic.delta - ic_cic.delta.mean()).ravel()
    r = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"[icdensity<->cic] z~99 correlation r = {r:.4f}")
    assert r > 0.8, f"linear vs CIC correlation unexpectedly low: r={r:.3f}"
