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
                     time=None, clustered=False, with_boxsize=True, positions=None,
                     with_icdensity=False, id_offset=0, shuffle=False, gap_ids=False,
                     velocities=None, use_peculiar_velocity=1):
    """Write a tiny IC bigfile. Particles on a regular ngrid^3 lattice at cell
    centres (=> uniform density), unless ``clustered`` or explicit ``positions``.

    With ``with_icdensity``, also write ``ID`` (row-major Lagrangian, +id_offset)
    and ``ICDensity`` = sin(2*pi*i/ngrid) (a known linear field varying along x)."""
    if positions is not None:
        pos = np.asarray(positions, "f8")
    else:
        g = (np.arange(ngrid) + 0.5) * box / ngrid
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype("f8")
        if clustered:
            pos = np.full_like(pos, box / 2.0)
    n = pos.shape[0]
    idx = np.arange(n)
    ii = idx // (ngrid * ngrid)
    icdensity = np.sin(2 * np.pi * ii / ngrid).astype("f4")
    ids = (idx + 1 + id_offset).astype("u8")
    if gap_ids:
        ids = ids.copy()
        ids[0] = ids.max() + 100            # non-contiguous: breaks max-min+1 == npart
    if shuffle:
        perm = np.random.RandomState(7).permutation(n)   # scramble on-disk order
        pos, ids, icdensity = pos[perm], ids[perm], icdensity[perm]
    with bigfile.File(str(path), create=True) as bf:
        for t in ptypes:
            bf.create_from_array(f"{t}/Position", pos)
            if velocities is not None:
                vel = np.asarray(velocities, "f4")
                if vel.ndim == 1:                      # broadcast a single (3,) vector
                    vel = np.tile(vel, (n, 1)).astype("f4")
                bf.create_from_array(f"{t}/Velocity", vel)
            if with_icdensity:
                bf.create_from_array(f"{t}/ID", ids)
                bf.create_from_array(f"{t}/ICDensity", icdensity)
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
        if use_peculiar_velocity is not None:
            h.attrs["UsePeculiarVelocity"] = np.array([use_peculiar_velocity], dtype="i4")
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


# --- ICDensity (linear delta_1) route -----------------------------------------
def test_icdensity_route_recovers_linear_field(tmp_path):
    # field="icdensity" reshapes the native ICDensity block by ID to the grid.
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, with_icdensity=True)
    f = ic.load_ic_density(p, ptype="dm", nmesh=8, field="icdensity")
    assert f.delta.shape == (8, 8, 8)
    expected = np.sin(2 * np.pi * np.arange(8) / 8).astype(np.float32)   # varies along x (axis 0)
    np.testing.assert_allclose(f.delta[:, 0, 0], expected, atol=1e-6)
    assert f.delta.mean() == pytest.approx(0.0, abs=1e-6)
    assert f.meta["field"] == "icdensity"


def test_icdensity_route_handles_id_offset(tmp_path):
    # DM-like high ID block (offset) must still scatter correctly (id - id.min()).
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, with_icdensity=True, id_offset=8 ** 3)
    f = ic.load_ic_density(p, ptype="dm", nmesh=8, field="icdensity")
    expected = np.sin(2 * np.pi * np.arange(8) / 8).astype(np.float32)
    np.testing.assert_allclose(f.delta[:, 0, 0], expected, atol=1e-6)


def test_icdensity_downsamples_by_block_average(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, with_icdensity=True)
    f = ic.load_ic_density(p, ptype="dm", nmesh=4, field="icdensity")
    assert f.delta.shape == (4, 4, 4)
    # block-average of sin over pairs of x-planes
    fine = np.sin(2 * np.pi * np.arange(8) / 8)
    coarse = fine.reshape(4, 2).mean(axis=1)
    np.testing.assert_allclose(f.delta[:, 0, 0], coarse, atol=1e-6)


def test_icdensity_route_needs_blocks(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, with_icdensity=False)  # no ICDensity/ID
    with pytest.raises(ValueError):
        ic.load_ic_density(p, ptype="dm", nmesh=8, field="icdensity")


def test_invalid_field_raises(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8)
    with pytest.raises(ValueError):
        ic.load_ic_density(p, ptype="dm", nmesh=8, field="bogus")


def test_icdensity_uses_id_not_on_disk_order(tmp_path):
    # Scramble on-disk order: a positional loader would fail; ID-based scatter recovers it.
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, with_icdensity=True, shuffle=True)
    f = ic.load_ic_density(p, ptype="dm", nmesh=8, field="icdensity")
    expected = np.sin(2 * np.pi * np.arange(8) / 8).astype(np.float32)
    np.testing.assert_allclose(f.delta[:, 0, 0], expected, atol=1e-6)


def test_icdensity_incomplete_ids_raise(tmp_path):
    # A non-contiguous / partial ID set (cubic count but a gap) must error, not mis-index.
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, with_icdensity=True, gap_ids=True)
    with pytest.raises(ValueError):
        ic.load_ic_density(p, ptype="dm", nmesh=8, field="icdensity")


def test_icdensity_nmesh_gt_ngrid_raises(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, with_icdensity=True)
    with pytest.raises(ValueError):
        ic.load_ic_density(p, ptype="dm", nmesh=16, field="icdensity")


def test_icdensity_nmesh_not_divisor_warns(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, with_icdensity=True)
    with pytest.warns(Warning):
        ic.load_ic_density(p, ptype="dm", nmesh=5, field="icdensity")


# --- raw particle access (mesh with your own CIC) -----------------------------
def test_load_ic_particles_returns_raw_columns(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, with_icdensity=True)
    data, header = ic.load_ic_particles(p, ptype="dm",
                                        columns=("Position", "ICDensity", "ID"))
    assert data["Position"].shape == (8 ** 3, 3)
    assert data["ICDensity"].shape == (8 ** 3,)
    assert data["ID"].shape == (8 ** 3,)
    assert 0.0 <= data["Position"].min() and data["Position"].max() < 120000.0  # kpc/h
    assert header["box_mpc_h"] == pytest.approx(120.0)
    assert header["redshift"] == pytest.approx(99.0)
    # Omega0/OmegaLambda must be exposed (mirrors load_ic_density's meta dict) so
    # consumers (e.g. the real-IC linear-theory velocity test) can use the sim's
    # true cosmology instead of a hardcoded fallback.
    assert header["Omega0"] == pytest.approx(0.3)
    assert header["OmegaLambda"] == pytest.approx(0.7)


def test_load_ic_particles_subsample(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8)
    data, _ = ic.load_ic_particles(p, ptype="dm", columns=("Position",), subsample=4)
    assert data["Position"].shape[0] == 8 ** 3 // 4


def test_load_ic_particles_missing_column_raises(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, with_icdensity=False)
    with pytest.raises(ValueError):
        ic.load_ic_particles(p, ptype="dm", columns=("ICDensity",))


# --- velocity conversion (UsePeculiarVelocity dispatch) ------------------------
def test_velocity_peculiar_flag_returned_unchanged(tmp_path):
    """PRIYA case: UsePeculiarVelocity=1 => the block is already peculiar km/s."""
    v = np.array([3.0, -4.0, 12.0], dtype="f4")
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=4, time=0.01, velocities=v,
                         use_peculiar_velocity=1)
    data, header = ic.load_ic_particles(p, ptype="dm", columns=("Position", "Velocity"))
    np.testing.assert_allclose(data["Velocity"], np.tile(v, (4 ** 3, 1)), rtol=1e-6)
    assert header["use_peculiar_velocity"] == 1
    assert header["velocity_units"] == "km/s (peculiar)"
    assert header["scale_factor"] == pytest.approx(0.01)


def test_velocity_gadget2_flag_is_scaled_by_sqrt_a(tmp_path):
    """UsePeculiarVelocity=0 => stored is v/sqrt(a); loader must multiply by sqrt(a)."""
    v = np.array([3.0, -4.0, 12.0], dtype="f4")
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=4, time=0.01, velocities=v,
                         use_peculiar_velocity=0)
    data, header = ic.load_ic_particles(p, ptype="dm", columns=("Position", "Velocity"))
    np.testing.assert_allclose(data["Velocity"], np.tile(v * 0.1, (4 ** 3, 1)), rtol=1e-5)
    assert header["use_peculiar_velocity"] == 0


def test_velocity_raw_returns_stored_block(tmp_path):
    v = np.array([3.0, -4.0, 12.0], dtype="f4")
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=4, time=0.01, velocities=v,
                         use_peculiar_velocity=0)
    data, header = ic.load_ic_particles(p, ptype="dm", columns=("Velocity",), velocity="raw")
    np.testing.assert_allclose(data["Velocity"], np.tile(v, (4 ** 3, 1)), rtol=1e-6)
    assert header["velocity_units"] == "raw (as stored)"


def test_velocity_without_flag_raises(tmp_path):
    """Never guess the convention: no flag in the Header => refuse to convert."""
    v = np.array([3.0, -4.0, 12.0], dtype="f4")
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=4, time=0.01, velocities=v,
                         use_peculiar_velocity=None)
    with pytest.raises(ValueError, match="UsePeculiarVelocity"):
        ic.load_ic_particles(p, ptype="dm", columns=("Velocity",))
    # ...but 'raw' still works, since no conversion is implied.
    data, _ = ic.load_ic_particles(p, ptype="dm", columns=("Velocity",), velocity="raw")
    assert data["Velocity"].shape == (4 ** 3, 3)


def test_position_only_load_needs_no_velocity_flag(tmp_path):
    """A Position-only load must not be broken by a missing velocity flag."""
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=4, use_peculiar_velocity=None)
    data, _ = ic.load_ic_particles(p, ptype="dm", columns=("Position",))
    assert data["Position"].shape == (4 ** 3, 3)


def test_bad_velocity_kwarg_raises(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=4)
    with pytest.raises(ValueError, match="velocity must be"):
        ic.load_ic_particles(p, ptype="dm", velocity="nonsense")


# --- velocity mesh (CIC momentum / mean-velocity field) ------------------------
def test_velocity_mesh_uniform_velocity_is_recovered(tmp_path):
    """Every particle has the same v => every occupied cell must show exactly that v.
    This is the test that pins the momentum/count normalisation p/n."""
    v = np.array([10.0, -20.0, 5.0], dtype="f4")
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, time=0.01, velocities=v,
                         use_peculiar_velocity=1)
    vf = ic.load_ic_velocity_mesh(p, ptype="dm", nmesh=8, field="velocity")
    assert vf.v.shape == (3, 8, 8, 8)
    assert vf.meta["empty_cells"] == 0
    for a in range(3):
        np.testing.assert_allclose(vf.v[a], v[a], rtol=1e-5)


def test_velocity_mesh_momentum_conserves_total(tmp_path):
    """CIC weights sum to 1 per particle => sum(momentum mesh) == sum(v) over particles."""
    rng = np.random.RandomState(3)
    n = 8 ** 3
    v = (rng.randn(n, 3) * 30.0).astype("f4")
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, time=0.01, velocities=v,
                         use_peculiar_velocity=1)
    vf = ic.load_ic_velocity_mesh(p, ptype="dm", nmesh=8, field="momentum")
    assert vf.field == "momentum"
    for a in range(3):
        assert vf.v[a].sum() == pytest.approx(v[:, a].sum(), rel=1e-4)


def test_velocity_mesh_empty_cells_are_zero_and_reported(tmp_path):
    """Far more cells than particles => empty cells are 0 and counted, with a warning."""
    v = np.array([10.0, 0.0, 0.0], dtype="f4")
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=2, time=0.01, velocities=v,
                         use_peculiar_velocity=1)          # 8 particles
    with pytest.warns(UserWarning, match="empty"):
        vf = ic.load_ic_velocity_mesh(p, ptype="dm", nmesh=16, field="velocity")
    assert vf.meta["empty_cells"] > 0
    assert np.all(np.isfinite(vf.v))            # 0/0 must never leak a NaN
    # every empty cell is exactly zero in every component
    empty_mask = np.all(vf.v == 0.0, axis=0)
    assert empty_mask.sum() >= vf.meta["empty_cells"]


def test_velocity_mesh_uses_peculiar_units(tmp_path):
    """The mesh must go through the same flag dispatch as the particles."""
    v = np.array([10.0, 0.0, 0.0], dtype="f4")
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, time=0.01, velocities=v,
                         use_peculiar_velocity=0)           # stored = v/sqrt(a)
    vf = ic.load_ic_velocity_mesh(p, ptype="dm", nmesh=8, field="velocity")
    np.testing.assert_allclose(vf.v[0], 10.0 * 0.1, rtol=1e-4)   # x sqrt(0.01)
    assert vf.units == "km/s (peculiar)"


def test_velocity_mesh_bad_field_raises(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=4, velocities=np.zeros(3))
    with pytest.raises(ValueError, match="field must be"):
        ic.load_ic_velocity_mesh(p, ptype="dm", nmesh=4, field="nonsense")


def test_velocity_mesh_missing_block_raises(tmp_path):
    p = _make_ic_bigfile(tmp_path / "ic", ngrid=4)           # no Velocity written
    with pytest.raises(ValueError, match="Velocity"):
        ic.load_ic_velocity_mesh(p, ptype="dm", nmesh=4)


def test_velocity_mesh_chunked_matches_unchunked(tmp_path):
    """Accumulating velocity across multiple chunks gives the same result as
    a single large chunk. Verifies the accumulator is not reset per chunk."""
    # Create random per-particle velocities (uniform would not catch reset bugs)
    rng = np.random.RandomState(42)
    n = 8 ** 3
    v_data = (rng.randn(n, 3) * 30.0).astype("f4")

    p = _make_ic_bigfile(tmp_path / "ic", ngrid=8, time=0.01, velocities=v_data,
                         use_peculiar_velocity=1)

    # Load with small chunk_size (forces many iterations)
    chunked_vel = ic.load_ic_velocity_mesh(p, ptype="dm", nmesh=8, field="velocity",
                                           chunk_size=37)
    # Load with large chunk_size (single or few iterations)
    full_vel = ic.load_ic_velocity_mesh(p, ptype="dm", nmesh=8, field="velocity",
                                        chunk_size=10 ** 9)
    np.testing.assert_allclose(chunked_vel.v, full_vel.v, rtol=1e-5)

    # Also verify momentum field gives same result
    chunked_mom = ic.load_ic_velocity_mesh(p, ptype="dm", nmesh=8, field="momentum",
                                           chunk_size=37)
    full_mom = ic.load_ic_velocity_mesh(p, ptype="dm", nmesh=8, field="momentum",
                                        chunk_size=10 ** 9)
    np.testing.assert_allclose(chunked_mom.v, full_mom.v, rtol=1e-5)
