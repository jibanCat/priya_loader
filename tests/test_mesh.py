"""Tests for the streaming cloud-in-cell (CIC) mesh painter.

The painter deposits particle positions onto a periodic 3D mesh, memory-safely
(accumulate into a preallocated array across chunks). It is pure numpy — no
bigfile dependency — so it is fully exercised here.
"""
import numpy as np
import pytest

from priya_loader import mesh


def test_integer_position_deposits_in_one_cell():
    # A particle exactly on a grid node (mesh units) puts all its mass in that cell.
    rho = mesh.cic_paint(np.array([[1.0, 2.0, 3.0]]), nmesh=4)
    assert rho.shape == (4, 4, 4)
    assert rho[1, 2, 3] == pytest.approx(1.0)
    assert rho.sum() == pytest.approx(1.0)


def test_half_cell_splits_into_eight():
    # A particle at a cell corner centre splits equally over 8 neighbours.
    rho = mesh.cic_paint(np.array([[0.5, 0.5, 0.5]]), nmesh=4)
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                assert rho[i, j, k] == pytest.approx(0.125)
    assert rho.sum() == pytest.approx(1.0)


def test_mass_is_conserved():
    rng = np.random.RandomState(0)
    pos = rng.uniform(0, 8, size=(1000, 3))
    rho = mesh.cic_paint(pos, nmesh=8)
    assert rho.sum() == pytest.approx(1000.0)


def test_periodic_wrap_at_high_edge():
    # i=nmesh-1, d=0.7 -> 0.3 stays at nmesh-1, 0.7 wraps to cell 0.
    rho = mesh.cic_paint(np.array([[3.7, 0.0, 0.0]]), nmesh=4)
    assert rho[3, 0, 0] == pytest.approx(0.3)
    assert rho[0, 0, 0] == pytest.approx(0.7)
    assert rho.sum() == pytest.approx(1.0)


def test_boxsize_scales_positions():
    # boxsize given -> positions in [0, boxsize) map to mesh units.
    rho = mesh.cic_paint(np.array([[60.0, 0.0, 0.0]]), nmesh=4, boxsize=120.0)
    # 60/120*4 = 2.0 -> integer cell 2
    assert rho[2, 0, 0] == pytest.approx(1.0)


def test_accumulate_into_out_equals_single_shot():
    rng = np.random.RandomState(1)
    pos = rng.uniform(0, 8, size=(500, 3))
    single = mesh.cic_paint(pos, nmesh=8)
    out = np.zeros((8, 8, 8))
    mesh.cic_paint(pos[:200], nmesh=8, out=out)
    mesh.cic_paint(pos[200:], nmesh=8, out=out)
    np.testing.assert_allclose(out, single, rtol=1e-12)


def test_overdensity_uniform_is_zero():
    # one particle per cell on the grid -> uniform density -> delta ~ 0.
    g = np.arange(4)
    pos = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T.astype(float)
    rho = mesh.cic_paint(pos, nmesh=4)
    delta = mesh.to_overdensity(rho)
    np.testing.assert_allclose(delta, 0.0, atol=1e-12)
    assert delta.mean() == pytest.approx(0.0, abs=1e-12)


def test_overdensity_clustered_is_positive_where_clustered():
    pos = np.tile([1.0, 1.0, 1.0], (50, 1))   # all in one cell
    rho = mesh.cic_paint(pos, nmesh=4)
    delta = mesh.to_overdensity(rho)
    assert delta[1, 1, 1] > 10
    assert delta.min() == pytest.approx(-1.0)   # empty cells are delta = -1


def test_wrap_at_boxsize_physical_units():
    # A particle at exactly Position == BoxSize wraps to cell 0 (periodic).
    rho = mesh.cic_paint(np.array([[120.0, 120.0, 120.0]]), nmesh=4, boxsize=120.0)
    assert rho[0, 0, 0] == pytest.approx(1.0)
    assert rho.sum() == pytest.approx(1.0)


def test_coarsen_ngrid_gt_nmesh_conserves_mass():
    # Production path: a fine particle lattice painted onto a coarser mesh.
    g = (np.arange(16) + 0.5) / 16.0 * 8.0       # 16^3 particles in mesh units [0,8)
    pos = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    rho = mesh.cic_paint(pos, nmesh=8)
    assert rho.sum() == pytest.approx(16 ** 3)
    assert np.isfinite(mesh.to_overdensity(rho)).all()


def test_out_must_be_float64_and_right_shape():
    pos = np.array([[1.0, 1.0, 1.0]])
    with pytest.raises((ValueError, TypeError)):
        mesh.cic_paint(pos, nmesh=4, out=np.zeros((4, 4, 4), dtype=np.float32))
    with pytest.raises((ValueError, TypeError)):
        mesh.cic_paint(pos, nmesh=4, out=np.zeros((3, 3, 3), dtype=np.float64))


def _explicit_cic(pos, nmesh, boxsize=None):
    """Transparent, obviously-correct CIC (explicit per-particle 8-corner loop) —
    the independent reference our vectorized bincount painter must match exactly."""
    rho = np.zeros((nmesh, nmesh, nmesh))
    p = np.asarray(pos, float)
    if boxsize is not None:
        p = p * (nmesh / boxsize)
    for x in p:
        i = np.floor(x).astype(int)
        d = x - i
        for ox in (0, 1):
            wx = (1 - d[0]) if ox == 0 else d[0]
            for oy in (0, 1):
                wy = (1 - d[1]) if oy == 0 else d[1]
                for oz in (0, 1):
                    wz = (1 - d[2]) if oz == 0 else d[2]
                    rho[(i[0] + ox) % nmesh, (i[1] + oy) % nmesh, (i[2] + oz) % nmesh] += wx * wy * wz
    return rho


def test_cic_matches_explicit_reference():
    # Proves the bincount vectorization == textbook CIC, bit-for-bit (no nbodykit needed).
    rng = np.random.RandomState(3)
    pos = rng.uniform(0, 16, size=(3000, 3))
    np.testing.assert_allclose(mesh.cic_paint(pos, 16), _explicit_cic(pos, 16), atol=1e-10)
    # with a physical boxsize too
    posb = rng.uniform(0, 120.0, size=(1500, 3))
    np.testing.assert_allclose(
        mesh.cic_paint(posb, 12, boxsize=120.0), _explicit_cic(posb, 12, boxsize=120.0), atol=1e-10
    )


def test_cic_identical_to_nbodykit():
    """Our raw CIC == nbodykit's UNcompensated, NON-interlaced CIC (the 'tricks'
    we deliberately omit). Auto-skips where nbodykit isn't installed."""
    nbk = pytest.importorskip("nbodykit")              # skip if absent
    from nbodykit.source.catalog import ArrayCatalog
    rng = np.random.RandomState(5)
    box, nmesh = 120.0, 16
    pos = rng.uniform(0, box, size=(5000, 3))
    ours = mesh.to_overdensity(mesh.cic_paint(pos, nmesh, boxsize=box))
    cat = ArrayCatalog({"Position": pos.astype("f8")})
    m = cat.to_mesh(Nmesh=nmesh, BoxSize=box, resampler="cic",
                    compensated=False, interlaced=False)
    theirs = np.asarray(m.compute(mode="real")) - 1.0   # nbodykit paints 1+delta
    np.testing.assert_allclose(ours, theirs, atol=1e-6)
