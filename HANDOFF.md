# Hand-off notes

A short orientation for using `priya_loader` to build the
`[params, redshift, IC δ₁, τ]` dataset for the Lyman-α flux-bias program. Read the
[README](README.md) for the full API and [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb)
for a runnable end-to-end demo.

## What this gives you, per (simulation, redshift)

```python
from priya_loader import PriyaDataset
ds = PriyaDataset(ROOT, fidelity="lowres", ic_nmesh=512, ic_field="icdensity", flux_axis=1)
for s in ds:                      # lazy: one (sim, z) in memory at a time
    s.params       # SimParams (cosmology + thermal/astro; authoritative box/npart)
    s.redshift     # float (from the τ file header)
    s.ic           # (nmesh, nmesh, nmesh) float32 linear δ₁ (real space, z_init≈99)
    s.tau          # (480, 480, nbins) float32 raw optical depth (redshift space)
ds.export("out/")                 # one .npz per (sim, z) for the JAX pipeline
```

## Recommended path for the bias cross-spectrum `b_F = P_{F,δ₁}/P_{δ₁δ₁}`

1. **Use the linear `δ₁`**: `ic_field="icdensity"` (the native `ICDensity` block,
   reshaped to the Lagrangian grid) — not the CIC field. `nmesh` **must divide**
   `ngrid` (1536 → 256/384/512; 3072 → 256/384/512/768). **Do not use `nmesh=480`
   with `icdensity`** (480 divides neither grid → uneven window → biased `β_F`).
2. **Growth-rescale** the z=99 amplitude to the flux redshift:
   `units.growth_factor(z_flux, Ω_m, Ω_Λ) / units.growth_factor(z_init, …)`
   (`Ω0/ΩΛ/Time` are in `s.params` / the npz `params`/`meta`).
3. **Co-register with τ** (anisotropic `(480,480,nbins)`, LOS = redshift-space
   velocity): transpose `δ₁` to `s.meta["tau_cube_axes"]`; the τ transverse grid is
   480, so **Fourier-match the common low-`k` modes** (or Fourier-resample your
   512 `δ₁` to 480) — both fields share the box origin (CIC node and `cofm` both at
   `j·box/N`). Real-space `δ₁` × redshift-space τ is the intended Kaiser estimator
   (gives `b_δ(1+β_F μ²)`; Kaiser 1987, MNRAS 227, 1; for the Lyα flux form see
   McDonald 2003, arXiv:astro-ph/0108064 and Slosar et al. 2011, arXiv:1104.5244);
   do **not** add RSD to `δ₁`.
4. **Window**: deconvolve the IC window for finite-`k` work (CIC `sinc²(k_iΔ/2)`,
   Jing 2005 arXiv:astro-ph/0409240; or the `icdensity` block-average `sinc`); it
   → 1 at the low-`k` modes that set `b_F`. (Pylians/nbodykit CIC conventions are
   pinned in `PROVENANCE.md`.)
5. **Flux**: `F = exp(−τ)`, `δ_F = F/F̄ − 1`. Choose `F̄` (raw box mean vs an
   observational/τ₀ target — sets the `b_F` amplitude) and decide on **DLA masking**
   (raw τ keeps saturated `τ>1e6` troughs, which bias large-scale `b_F`).

If you'd rather mesh yourself, `load_ic_particles(...)` returns the raw particle
columns. Our CIC is verified bit-for-bit vs an explicit reference and vs Pylians;
we ship raw (no nbodykit compensation/interlacing).

## Data staging — check before you run

- **Production ICs may not be staged.** The IC bigfile is `ICS/120_<Ngrid>_99`;
  on a partially-transferred tree it can be an empty skeleton (then `s.ic` is
  `None` and you get τ-only samples). Verify a real IC reads before a big run:
  `load_ic_density(ic_dir, field="icdensity", nmesh=512)`.
- **τ may be partially staged in redshift.** `find_spectra_files(sim_dir)` reports
  what's present; the loader skips missing/empty/truncated files with a warning.
- Box/Ngrid are taken from `mpgadget.param`/`_genic_params.ini`, **not** the
  `SimulationICs.json` (whose `box`/`npart` read a stale `15`/`192` for ~half the
  suite). This is handled for you.

## Caveats

- τ is **redshift-space only**. Separating `b_δ` from `b_η` cleanly would need a
  *regenerated* real-space `fake_spectra` run (not shipped).
- `ic_field="icdensity"` requires the `ICDensity` + `ID` blocks in the IC bigfile.
- Login-node memory: `ic_nmesh ≤ 512` (see the README memory table); one τ axis is
  ~1.5 GB. The IC read is I/O-bound (81 GB/type at 1536³, 648 at 3072³).

## Provenance

Every external convention (units, `Position`/`ICDensity` definitions, the τ schema
+ axis ordering, the folder-name + `Ap` pivot, CLASS-not-CAMB) is traced to a
git-stamped upstream source in [`PROVENANCE.md`](PROVENANCE.md), with raw/blob URLs
so you can re-check any claim.
