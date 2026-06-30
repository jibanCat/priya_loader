# Changelog

All notable changes to `priya_loader`. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow semver.

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
  bit-for-bit vs an explicit reference and vs Pylians) and `field="icdensity"`
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
