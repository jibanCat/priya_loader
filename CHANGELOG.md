# Changelog

All notable changes to `priya_loader`. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow semver.

## [Unreleased]

### Added
- `load_ic_velocity_mesh` — CIC velocity/momentum field from the IC particles
  (`(3, nmesh, nmesh, nmesh)`, peculiar km/s, co-registered with `load_ic_density`);
  `ICVelocityField`.
- `units.ic_velocity_to_peculiar_kms` — converts a stored IC `Velocity` block to
  peculiar km/s by dispatching on the IC header's `UsePeculiarVelocity` flag.
- `notebooks/ic_particles.ipynb` — short demo of DM positions/velocities.
- Real-IC test asserting `v_rms` matches linear theory `a·H·f·σ_disp` (catches a
  `sqrt(a)` units error, which is a 10× effect at z=99).
- `tests/test_real_ic.py` — real-IC verification, gated behind `PRIYA_REAL_IC`
  (skips by default). Codifies the NERSC sign-off the synthetic fixtures can't
  reach: resolution/box invariants, the linear `δ₁` (`icdensity`) and Eulerian
  `cic` sanity, **CIC equivalence on real particle positions** — against a built-in
  dependency-free reference (runs unconditionally) plus optional Pylians (in a
  `numpy<2` env) and nbodykit cross-checks — and `icdensity`↔`cic` consistency at
  z≈99. Verified on NERSC against the staged **512³** low-res companion
  (`120_512_99`) and the **3072³** hi-res IC (resolution/box invariants only — a
  full 3072³ particle CIC would read ~700 GB); the 1536³ low-res grid referenced in
  older notes is not staged there.
- `[notebook]` install extra (`jupyterlab` + `ipykernel` + `matplotlib` + `bigfile`)
  so `notebooks/quickstart.ipynb` runs from a clean install.
- Provenance: expanded git-stamped citations in `PROVENANCE.md` / `HANDOFF.md` and inline
  in the notebook (units, cosmology helpers, the `ID`-decode formula, Pylians/nbodykit/
  bigfile pins); corrected the `A_p` pivot label (8 Mpc scale, `k≈0.785 Mpc⁻¹`).

### Changed
- **`load_ic_particles` now returns `Velocity` in peculiar km/s by default**
  (`velocity="peculiar_kms"`; pass `velocity="raw"` for the stored block). It reads
  `UsePeculiarVelocity` from the IC header and **raises** if a conversion is
  requested and the flag is absent. For PRIYA the flag is 1, so the *values* are
  unchanged — what is new is the guarantee and the metadata (`scale_factor`,
  `use_peculiar_velocity`, `velocity_units` in the returned header).
- `notebooks/quickstart.ipynb` rewritten as a step-by-step **pedagogical** tutorial
  for users new to MP-Gadget: it defaults to the staged NERSC PRIYA path, makes
  explicit that the low-res ICs are **512³**, and walks through loader usage plus
  simple matplotlib visualizations of the parameters, τ/flux fields, and IC density.

## [0.1.0] — 2026-06-29

First tagged release. ML-friendly access to the PRIYA MP-Gadget Lyman-α suite.

### Added
- **`PriyaDataset`** — the headline orchestrator: loops over every
  (simulation, redshift) and yields `Sample(params, redshift, ic, tau)`
  (`.as_tuple()` for the bare tuple); lazy, login-node-safe, graceful on
  partial/mid-transfer data; `.export()` writes one self-describing `.npz` per
  (sim, z).
- **`load_tau_grid`** — raw, redshift-space Lyman-α optical depth as a
  `(ngrid, ngrid, nbins)` cube per LOS axis (h5py only); `cube_axes`, `space`,
  `dv_kms` metadata; `to_flux`/`mean_flux`/`to_delta_flux` helpers.
- **`load_ic_density`** — IC density mesh from the MP-GenIC bigfile, two fields:
  `field="cic"` (Eulerian CIC of displaced particles; built-in numpy CIC verified
  bit-for-bit vs an explicit reference, and to float32 roundoff vs Pylians) and `field="icdensity"`
  (the native linear `δ₁` on the Lagrangian grid — the bias-cross-spectrum
  reference; `nmesh` must divide `ngrid`).
- **`load_ic_particles`** — raw particle columns (Position/Velocity/ICDensity/ID),
  no meshing, for painting with your own nbodykit/Pylians.
- **`SimParams`** — parameters from the *authoritative* run config
  (`mpgadget.param`/`_genic_params.ini`), not the often-stale `SimulationICs.json`
  `box`/`npart`; folder-name parsing; resolution + amplitude-pivot validation.
- **`units`** — MP-Gadget unit system + `growth_factor`/`growth_rate`,
  `hubble_z`/`velfac`, particle mass.
- `runconfig`, `paths` (discovery, robust to partial staging), `mesh` (streaming
  CIC painter).
- **`PROVENANCE.md`** — every external convention traced to a git-stamped upstream
  source (MP-Gadget, fake_spectra, lya_emulator, SimulationRunner), independently
  verified to resolve at the pinned commits.
- `notebooks/quickstart.ipynb` — reproducible tutorial (tiny synthetic fixtures).
- Packaging: minimal numpy+h5py core; `[ic]`/`[plots]`/`[dev]` extras;
  `MANIFEST.in`; `CITATION.cff`.

### Notes
- Only the **redshift-space** τ is shipped (real-space would need a regenerated
  `fake_spectra` run).
- The IC path is verified on synthetic bigfiles + a reference CIC; a real-IC
  smoke test on a staged 1536³ bigfile is recommended (see `HANDOFF.md`).
