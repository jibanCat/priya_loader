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

All tests here use ``ptype="dm"`` — which is the **CDM** field, not total matter
(each species carries its own CLASS transfer function; see ``ic.py``). The gas
(type-0) route is code-symmetric (``PTYPE["gas"]=0``) and is covered on synthetic
fixtures; it is not separately asserted against a real IC.

The two ``@pytest.mark.slow`` tests at the bottom are the **cosmology guards**: they
compare P(k) measured from the loaders against ``ICS/inputspec_*.txt`` — the linear
spectrum MP-GenIC was actually handed — so any amplitude error (a stray sqrt(a), a
missing 1/h, a bad growth rate, a mis-normalised momentum field) fails a test rather
than quietly biasing someone's b_F. Each streams the full particle block (~5 min).
"""
import os

import numpy as np
import pytest

from priya_loader import (load_ic_density, load_ic_particles, load_ic_velocity_mesh,
                          mesh, units)
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


@pytest.fixture(scope="module")
def real_ic_particles_posvel():
    """Position + Velocity + ID from the real IC (strided subsample), plus ngrid."""
    _skip_if_heavy_load()
    data, header = load_ic_particles(
        REAL_IC, ptype="dm", columns=("Position", "Velocity", "ID"),
        subsample=SUBSAMPLE,
    )
    # ngrid from the FULL block size, not the subsample (subsample strides it down).
    bigfile = pytest.importorskip("bigfile")
    with bigfile.File(str(REAL_IC)) as bf:
        ngrid = _cube_root(int(bf["1/Position"].size))
    header["ids"] = data["ID"]
    return data["Position"].astype("f8"), data["Velocity"].astype("f8"), header, ngrid


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


# --- velocity physics (the check that catches a sqrt(a) units error) ---------
def test_real_ic_velocity_matches_linear_theory(real_ic_particles_posvel):
    """v_rms must equal the linear-theory prediction a*H(a)*f * sigma_disp.

    Zel'dovich: v = a * H(a) * f * Psi, with Psi = x - q the displacement. So the
    rms speed is fixed by the rms displacement measured from the SAME IC — no
    external normalisation, no power spectrum needed.

    This is the test with teeth: applying the Gadget-2 sqrt(a) factor to PRIYA's
    (already-peculiar) velocities makes v_rms 10x too small at z=99 and fails here.

    PRIYA sets ScaleDepVelocity (= DifferentTransferFunctions = 1), so v is drawn
    from the CLASS velocity transfer function rather than being exactly f*a*H*Psi;
    hence the generous tolerance. A units error is a factor of 10 — nowhere near it.

    Uses the real sim's Omega0/OmegaLambda from the IC Header (load_ic_particles'
    header now carries them, mirroring load_ic_density's meta dict). PRIYA varies
    Omega_m across sims (~0.25-0.34), and the prediction scales as sqrt(Omega_m)
    through H(z), so a hardcoded constant would be off by up to ~17% against this
    test's 20% tolerance — nearly insensitive to the very bug it targets. If the
    Header lacks Omega0 we skip rather than substitute a wrong cosmology.
    """
    pos, vel, header, ngrid = real_ic_particles_posvel
    box_kpc_h = header["box_kpc_h"]
    a = header["scale_factor"]
    z = header["redshift"]
    om = header.get("Omega0")
    ol = header.get("OmegaLambda")
    hubble = header.get("hubble")
    if not om or ol is None:
        pytest.skip(
            "IC Header has no Omega0/OmegaLambda; cannot verify against true "
            "cosmology (no fallback — see docstring)"
        )
    if not hubble:
        pytest.skip(
            "IC Header has no hubble (h); cannot convert sigma_disp from "
            "kpc/h to physical Mpc for the hubble_z(km/s/Mpc) prediction "
            "(no fallback — see docstring)"
        )

    # Displacement Psi = Position - q, with q the Lagrangian site decoded from ID
    # (same convention as ic._load_icdensity_grid: lag = (ID-1) % npart).
    ids = header["ids"].astype(np.int64)
    lag = (ids - 1) % (ngrid ** 3)
    q = np.stack([lag // (ngrid * ngrid),
                  (lag // ngrid) % ngrid,
                  lag % ngrid], axis=1) * (box_kpc_h / ngrid)
    d = pos - q
    d -= box_kpc_h * np.round(d / box_kpc_h)            # periodic minimum image
    # MP-GenIC writes Pos = q + shift(species) + Psi: each species is offset from the
    # Lagrangian grid by a CONSTANT (genic/main.c:63-64, shift_dm = 0.5*Omega_b/Omega0
    # * meanspacing = 18.05 kpc/h here). That shift is not part of Psi and carries no
    # velocity, so leaving it in inflates sigma_disp (128.5 -> 124.9 kpc/h once removed)
    # and biases the ratio low. Subtract the per-axis mean to drop it.
    d -= d.mean(axis=0)
    sigma_disp = float(np.sqrt((d ** 2).sum(axis=1).mean()))      # kpc/h, 3D rms

    # a*H(a)*f * Psi. hubble_z returns H in km/s/Mpc (physical Mpc; h is already
    # baked in via the 100.0*hubble prefactor — see units.hubble_z docstring).
    # sigma_disp is comoving kpc/h, so convert to comoving Mpc (h divided out,
    # matching hubble_z's physical-Mpc units) via /1000.0/hubble — same pattern
    # as units.velfac and tau.dv_kms.
    H = units.hubble_z(z, om, ol, hubble)                         # km/s/Mpc
    f = units.growth_rate(z, om, ol)
    predicted = a * H * f * (sigma_disp / 1000.0 / hubble)        # kpc/h -> comoving Mpc
    measured = float(np.sqrt((vel ** 2).sum(axis=1).mean()))

    assert measured == pytest.approx(predicted, rel=0.20), (
        f"v_rms={measured:.3f} km/s vs linear-theory {predicted:.3f} km/s "
        f"(ratio {measured / predicted:.3f}). A ratio near sqrt(a)={np.sqrt(a):.3f} "
        f"or 1/sqrt(a) means the UsePeculiarVelocity dispatch is wrong."
    )


def test_real_ic_velocity_is_peculiar_kms(real_ic_particles_posvel):
    """PRIYA ICs must declare UsePeculiarVelocity=1 (GenIC's default; the ini
    doesn't override it). If this ever flips, the loader adapts — but we want to
    KNOW, because every doc we ship states flag=1."""
    _, _, header, _ = real_ic_particles_posvel
    assert header["use_peculiar_velocity"] == 1
    assert header["velocity_units"] == "km/s (peculiar)"


def test_real_ic_gas_and_dm_velocities_differ():
    """ScaleDepVelocity (= DifferentTransferFunctions = 1 in PRIYA) means gas and
    DM are drawn from DIFFERENT velocity transfer functions. If these came back
    identical, our 'ptype matters' claim in the docs would be false."""
    _skip_if_heavy_load()
    kw = dict(columns=("Velocity",), subsample=SUBSAMPLE)
    try:
        gas, _ = load_ic_particles(REAL_IC, ptype="gas", **kw)
    except ValueError as e:
        pytest.skip(f"no gas block in this IC: {e}")
    dm, _ = load_ic_particles(REAL_IC, ptype="dm", **kw)
    n = min(len(gas["Velocity"]), len(dm["Velocity"]))
    assert not np.allclose(gas["Velocity"][:n], dm["Velocity"][:n]), (
        "gas and DM IC velocities are identical — ScaleDepVelocity assumption is wrong"
    )


# --- cosmology normalisation: P(k) vs the linear spectrum GenIC was handed --------
def _binned_pk(field, nmesh, box_mpc_h):
    """Binned P(k) [(Mpc/h)^3] of a real field on a periodic nmesh^3 box."""
    f = np.asarray(field, dtype=np.float32)
    f = f - f.mean()
    F = np.fft.rfftn(f)
    kf = 2 * np.pi / box_mpc_h
    kx = np.fft.fftfreq(nmesh, d=1.0 / nmesh)[:, None, None] * kf
    ky = np.fft.fftfreq(nmesh, d=1.0 / nmesh)[None, :, None] * kf
    kz = np.fft.rfftfreq(nmesh, d=1.0 / nmesh)[None, None, :] * kf
    kk = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    P = (np.abs(F) ** 2) * (box_mpc_h ** 3 / nmesh ** 6)
    bins = np.logspace(np.log10(kf), np.log10(0.5 * np.pi * nmesh / box_mpc_h), 16)
    idx = np.digitize(kk.ravel(), bins)
    kr, Pr = kk.ravel(), P.ravel()
    ks, Ps = [], []
    for b in range(1, len(bins)):
        m = idx == b
        if m.sum() > 8:
            ks.append(kr[m].mean())
            Ps.append(Pr[m].mean())
    return np.array(ks), np.array(Ps)


def _sim_dir():
    """<sim>/ from <sim>/ICS/<box>_<ngrid>_99 — the inputspec/transfer files live there."""
    return os.path.dirname(os.path.dirname(os.path.abspath(REAL_IC)))


def _input_pk():
    """The linear P(k) MP-GenIC was handed: ICS/inputspec_<base>.txt.
    genic/main.c: col1 = k [h/Mpc], col2 = DeltaSpec(k, DELTA_TOT)^2 in (kpc/h)^3."""
    base = os.path.basename(os.path.abspath(REAL_IC))
    path = os.path.join(_sim_dir(), "ICS", f"inputspec_{base}.txt")
    if not os.path.exists(path):
        pytest.skip(f"no inputspec file at {path}")
    k, p = np.loadtxt(path, unpack=True)
    return k, p / 1e9                       # (kpc/h)^3 -> (Mpc/h)^3


@pytest.mark.slow
def test_real_ic_velocity_power_matches_the_input_spectrum():
    """THE cosmology guard: -div(v)/(aHf) must reproduce the linear P(k) GenIC was given.

    In linear theory v(k) = i (k/k^2) a H f delta_1(k), so -div(v)/(aHf) IS delta_1 and
    its power spectrum must equal the input spectrum. This validates the WHOLE units
    chain at once — the UsePeculiarVelocity dispatch, the sqrt(a) convention, the h
    factor in hubble_z, and the growth rate. Any of them wrong and the amplitude moves:
    a stray sqrt(a) -> ratio 0.01; a missing 1/h -> ~0.55. Measured: 1.002.

    Slow: streams the full particle block once (~5 min on the 512^3 IC).
    """
    _skip_if_heavy_load()   # a 3072^3 target is ~1 TB of I/O — never on a shared login node
    nmesh = 128
    vf = load_ic_velocity_mesh(REAL_IC, ptype="dm", nmesh=nmesh, field="velocity")
    om, ol, h = vf.meta["Omega0"], vf.meta["OmegaLambda"], vf.meta["hubble"]
    if om is None or ol is None or h is None:
        pytest.skip("IC Header lacks the cosmology needed to predict the amplitude")
    z = vf.redshift
    a = 1.0 / (1.0 + z)
    aHf = a * units.hubble_z(z, om, ol, h) / h * units.growth_rate(z, om, ol)

    box = vf.box
    kf = 2 * np.pi / box
    kx = np.fft.fftfreq(nmesh, d=1.0 / nmesh)[:, None, None] * kf
    ky = np.fft.fftfreq(nmesh, d=1.0 / nmesh)[None, :, None] * kf
    kz = np.fft.rfftfreq(nmesh, d=1.0 / nmesh)[None, None, :] * kf
    v = vf.v.astype(np.float32)
    div = np.fft.irfftn(
        1j * (kx * np.fft.rfftn(v[0]) + ky * np.fft.rfftn(v[1]) + kz * np.fft.rfftn(v[2])),
        s=(nmesh, nmesh, nmesh),
        axes=(0, 1, 2),
    )
    theta = -div / aHf                      # = delta_1 in linear theory

    k, P = _binned_pk(theta, nmesh, box)
    kin, pin = _input_pk()
    sel = (k > 2 * kf) & (k < 0.8)          # above the fundamental, below the CIC window
    ratio = float(np.mean(P[sel] / np.interp(k[sel], kin, pin)))
    assert ratio == pytest.approx(1.0, abs=0.10), (
        f"P[-div(v)/aHf] / P_input = {ratio:.3f}, expected ~1.0. The IC velocity "
        f"normalisation is off — check the UsePeculiarVelocity dispatch (a stray "
        f"sqrt(a) gives ~0.01) and the h factor in hubble_z (a missing 1/h gives ~0.55)."
    )


@pytest.mark.slow
def test_real_ic_dm_density_power_is_cdm_not_total_matter():
    """delta_1(ptype="dm") is the CDM field: ~9% MORE power than total matter.

    This is species physics, not a bug — but it is a 4.6% amplitude offset that would
    propagate into b_F if a consumer normalised this field against a total-matter P(k).
    We pin it against the sim's OWN CLASS transfer functions so that (a) the docs cannot
    drift from the data, and (b) an actual amplitude regression in the icdensity route
    still fails the test.

    Slow: streams the full particle block once.
    """
    _skip_if_heavy_load()   # a 3072^3 target is ~1 TB of I/O — never on a shared login node
    nmesh = 128
    df = load_ic_density(REAL_IC, ptype="dm", nmesh=nmesh, field="icdensity")
    k, P = _binned_pk(df.delta, nmesh, df.box)
    kin, pin = _input_pk()

    tpath = os.path.join(_sim_dir(), "camb_linear", "ics_transfer_99.dat")
    if not os.path.exists(tpath):
        pytest.skip(f"no transfer file at {tpath}")
    t = np.loadtxt(tpath)
    if t.shape[1] < 6:
        pytest.skip("unexpected transfer-file layout")
    # no massive neutrinos in PRIYA => 16 cols: k, d_g, d_b, d_cdm, d_ur, d_tot, ...
    tk, d_b, d_cdm = t[:, 0], t[:, 2], t[:, 3]
    om, ob = df.meta["Omega0"], df.meta.get("OmegaBaryon")
    if om is None or ob is None:
        pytest.skip("IC Header lacks Omega0/OmegaBaryon")
    fb = ob / om
    d_m = (1.0 - fb) * d_cdm + fb * d_b          # GenIC's DELTA_TOT is matter-only
    kf = 2 * np.pi / df.box
    sel = (k > 2 * kf) & (k < 1.0)
    measured = float(np.mean(P[sel] / np.interp(k[sel], kin, pin)))
    predicted = float(np.mean(np.interp(k[sel], tk, (d_cdm / d_m) ** 2)))

    assert measured == pytest.approx(predicted, rel=0.06), (
        f"P(delta_1|dm) / P_input = {measured:.3f}, but the sim's own CLASS transfer "
        f"functions predict (T_cdm/T_matter)^2 = {predicted:.3f}. Either the icdensity "
        f"amplitude has regressed, or the species assumption in the docs is wrong."
    )
    assert measured > 1.02, "expected the CDM field to carry MORE power than total matter"
