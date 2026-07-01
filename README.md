# priya_loader

ML-friendly access to the **PRIYA** Lyman-α simulation suite — a grid of
MP-Gadget cosmological hydrodynamic simulations. `priya_loader` turns each
simulation into clean 3-D numpy arrays so you can treat the suite as a dataset:

- the **initial-condition (IC) density field** (from the MP-GenIC `bigfile`
  ICs, painted onto a mesh), and
- the **Lyman-α optical-depth field** `τ` at each output redshift (the raw
  gridded `fake_spectra` `tau` skewers, read with `h5py` and returned
  *unmodified*). Flux `F = exp(−τ)` and `δ_F = F/⟨F⟩ − 1` are provided as
  optional derived helpers.

The headline object `PriyaDataset` loops over every simulation/redshift and
yields, per snapshot:

```python
[ SimParams, redshift, ic_density_3d, tau_3d ]
```

so a downstream pipeline (e.g. a JAX bias-estimation code) can `np.load` and go:

```python
from priya_loader import PriyaDataset

ds = PriyaDataset("/path/to/priya/emu_full", fidelity="lowres",
                  ic_nmesh=256, ic_field="cic", flux_axis=1)   # ic_field="icdensity" for linear δ₁
for s in ds:                       # lazy: one (sim, redshift) in memory at a time
    s.params, s.redshift, s.ic, s.tau     # SimParams, float, (nmesh³) δ, (480,480,nbins) τ
    # ... or s.as_tuple() -> (params, redshift, ic, tau)

ds.export("out/")                  # one .npz per (sim, z) for the JAX pipeline
```

It degrades gracefully on partial/mid-transfer data: sims with no staged `tau`
are skipped, a missing production IC yields `ic=None` (τ-only samples), and a
folder that fails parameter validation is skipped with a warning.

📓 **Tutorial:** [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) is a
step-by-step, beginner-friendly walk (no MP-Gadget experience assumed) through
params → τ/flux → IC density → `PriyaDataset` → `.npz` export, with simple
matplotlib visualizations. It is written against the **real** staged PRIYA tree and
defaults to the NERSC path — the low-res ICs are **512³** (`ICS/120_512_99`); edit
the `ROOT` cell to point at your own copy. To run it, `pip install -e ".[ic,notebook]"`
(adds bigfile + matplotlib + Jupyter). Note: `PriyaDataset`'s IC discovery targets
the **production** grid (1536³/3072³); where only the 512³ companion is staged, use
the explicit `load_ic_density(.../ICS/120_512_99, …)` shown in the notebook.

> **Status.** This version ships the full stack: the building blocks (`units`,
> `params`, `runconfig`, `paths`), the Lyman-α `tau` loader (`load_tau_grid`), the
> IC density loader (`load_ic_density`), **and the `PriyaDataset` orchestrator**
> that ties them into the `[params, z, ic, tau]` tuple.

## Install

Not on PyPI yet — clone and install editable. Lightweight core (flux fields +
exported `.npz`; runs on a NERSC login node, no MPI):

```bash
git clone https://github.com/jibanCat/priya_loader.git
cd priya_loader
pip install -e .            # core = numpy + h5py only
```

Add the initial-conditions path (pulls in `bigfile`), or the dev/plotting tools:

```bash
pip install -e ".[ic]"      # initial conditions
pip install -e ".[dev]"     # tests + matplotlib + bigfile
```

Or create the full contributor environment:

```bash
conda env create -f environment.yml && conda activate priya_loader && pip install -e .
```

An `nbodykit`-based meshing backend is **not yet implemented** (only the
pure-numpy CIC ships today); `load_ic_density(..., backend="nbodykit")` raises.

## Quick start

Paths are always explicit — nothing is hard-coded, so the same code runs on a
laptop, Great Lakes, or NERSC.

```python
from priya_loader import paths, params

ROOT = "/path/to/priya/emu_full"          # wherever the suite lives

for sim in paths.discover_simulations(ROOT):
    p = params.SimParams.from_dir(sim.directory)
    tau_files = paths.find_spectra_files(sim.directory)    # [(snap_index, path), ...]
    print(p.name, p.fidelity, f"box={p.box} Mpc/h", f"npart={p.npart}",
          f"z_range=[{p.z_end}, {p.z_init}]", f"tau_snaps={len(tau_files)}")
    feature_vector = p.as_vector(["ns", "scalar_amp", "hubble", "omega0", "bhfeedback"])
```

`npart` follows the suite convention: it is the **one-dimensional** particle/grid
count, so `npart=1536` means a 1536³ run. `box` (Mpc/h) and `npart` come from the
authoritative run files, with the source of each recorded on the `RunConfig`
(`box_source="_genic_params.ini"`, `ngrid_source="mpgadget.param"`).

### Loading the Lyman-α optical depth

```python
from priya_loader import load_tau_grid, to_flux

snap, path = tau_files[0]
g = load_tau_grid(path, axis=1)   # axis 1=x, 2=y, 3=z; reads ~1.5 GB float32 (one axis)
g.tau            # (480, 480, nbins) RAW optical depth, redshift-space (LOS = last/velocity axis)
g.redshift       # from the file header
g.cube_axes      # ('y','z','x') for axis=1 — physical coord of each cube index (LOS last)
flux = to_flux(g.tau)            # exp(-tau); allocates another ~1.5 GB
```

`tau` is returned **exactly as stored** (no mean-flux/τ₀ rescaling). The cube is
**anisotropic**: 250 ckpc/h transverse (`box/480`) vs ≈70–110 ckpc/h along the LOS
(`box/nbins`, z/cosmology-dependent), and the LOS pixel is ≈10 km/s (`g.meta["dv_kms"]`). Each axis cube
is ~1.5 GB (`230 400 × nbins` float32); the loader reads **one axis at a time** so
it fits a NERSC login node — all three axes would be ~4.6 GB.

### Loading the initial-condition density

```python
from priya_loader import load_ic_density

# Eulerian CIC density of the displaced particles (default):
f = load_ic_density(ic_dir, ptype="dm", nmesh=256)              # -> rho/<rho>-1
# OR the exact linear delta_1 for the bias cross-spectrum (de Belsunce et al.):
f = load_ic_density(ic_dir, ptype="dm", nmesh=512, field="icdensity")  # nmesh must divide ngrid
f.delta        # (nmesh, nmesh, nmesh) float32, axes (x,y,z)
f.redshift     # IC redshift (~99); f.meta has Omega0/OmegaLambda/Time for D(z)
```

**Choosing the field.** `field="icdensity"` returns the native **linear `δ₁`**
(the `ICDensity` block on the Lagrangian grid) — the exact field a field-level
bias estimator wants; `field="cic"` (default) returns the **Eulerian** density of
the displaced particles.

> ⚠️ **`icdensity` `nmesh` must divide `ngrid`** (1536 → 256/384/512; 3072 →
> 256/384/512/768). The block-average is exact only at `nmesh=ngrid` (modulo
> GenIC's ~1-cell Gaussian smoothing, sub-percent at `k≲1`) and a clean `sinc`
> top-hat at divisors. **Do NOT use `nmesh=480` with `icdensity`** — 480 divides
> neither 1536 nor 3072, so the bins are uneven and the window biases `β_F` at
> finite `k`. `field="cic"` takes **any** `nmesh` (use 480 to match τ). Both
> fields are **real-space**, at `z_init≈99`.

To cross-correlate with `tau` for the flux bias, you (the consumer) handle:
- the **growth rescale** `D(z_flux)/D(z_init)` (use `units.growth_factor`);
- **co-registration with the 480 τ transverse grid**: with `cic`, paint at
  `nmesh=480` and transpose to `tau.cube_axes`; with `icdensity`, load at a
  *divisor* `nmesh` (e.g. 512) and **cross in Fourier space over the common low-`k`
  modes** (or Fourier-resample) — never block-average to 480;
- the τ **LOS is a redshift-space velocity axis** — resample it separately;
- **deconvolving the IC window** (CIC `sinc²`, or the `icdensity` block-average `sinc`);
- the **mean-flux** normalization, and whether to **mask/fill DLAs** in raw τ
  (saturated `τ>1e6` troughs are density-correlated and bias large-scale `b_F`).

`bigfile` is required (`pip install -e ".[ic]"`).

**Third path — raw particles (mesh it yourself).** If you'd rather control the
mass assignment (e.g. paint with your own nbodykit/Pylians), get the raw columns —
no meshing:

```python
from priya_loader import load_ic_particles
data, header = load_ic_particles(ic_dir, ptype="dm", columns=("Position",), subsample=1)
data["Position"]        # (N, 3) comoving kpc/h ; also "Velocity"/"ICDensity"/"ID"
# header: box_mpc_h, hubble, redshift
```

(Our built-in CIC is verified bit-for-bit vs an explicit reference, and to float32 roundoff vs Pylians;
nbodykit's compensation/interlacing are opt-in "tricks" we don't apply.)

**IC memory vs `nmesh`** (one τ axis ≈ 1.46 GB; reading `Position` is I/O-bound:
81 GB/type at 1536³, 648 GB/type at 3072³):

| `nmesh` | IC paint peak | IC δ (f4) | per-sample (δ+τ) | login node? |
|--------:|--------------:|----------:|-----------------:|:-----------:|
| 256 | 1.3 GB | 0.06 GB | 1.5 GB | ✅ |
| 480 | 2.7 GB | 0.41 GB | 1.9 GB | ✅ (matches τ transverse — **`cic` only**) |
| 512 | 3.0 GB | 0.50 GB | 2.0 GB | ✅ |
| 1024 | 17 GB | 4.0 GB | 5.5 GB | ⚠️ tight |
| 1536 | 55 GB | 13.5 GB | 15.0 GB | ❌ batch node |

Peak ≈ `2·nmesh³` float64 (mesh + one transient bincount) + ~1 GB chunk; **use
`nmesh ≤ 512` on a login node** (for `cic`, `480` matches the τ transverse grid;
for `icdensity` use a divisor of `ngrid` — see the warning above).

### Why a `runconfig` module?

The per-simulation `SimulationICs.json` carries **stale template `box`/`npart`**
for a large fraction of the suite (it reads `box=15, npart=192` even for
production 120 Mpc/h, 1536³ runs). `priya_loader` therefore reads the box and
particle grid from the authoritative run files (`mpgadget.param`,
`_genic_params.ini`) and uses the JSON only for cosmology/astrophysics. This is
handled for you in `SimParams.from_dir`.

## Data scope

**Simulations.** `emu_full` is the low-res suite (**1536³ particles, 120 Mpc/h**,
~60 simulations); `emu_full_hires` is the high-res suite (**3072³, 120 Mpc/h**).
Each simulation is one point in a 9-dimensional Latin hypercube. The ranges below
are **measured from the 60 staged `emu_full` folders** (via `SimParams`), not
quoted from a paper:

| group | parameter | symbol | range (emu_full) |
|-------|-----------|--------|------------------|
| cosmology | `ns` | n_s | 0.80 – 1.04 |
| cosmology | `Ap` (folder) / `scalar_amp` | A_P(k=0.78) | 1.2 – 2.6 ×10⁻⁹ |
| cosmology | `hubble` | h | 0.65 – 0.75 |
| cosmology | `omega_m_h2` | Ω_m h² | 0.140 – 0.146 |
| HeII reion. | `here_i`, `here_f`, `alpha_q` | z_i, z_f, α_q | 3.5–4.5, 2.2–3.2, 1.3–3.0 |
| HI reion. | `hireionz` | z_HI | 6.5 – 8.0 |
| feedback | `bhfeedback` | ε_BH | 0.031 – 0.069 |

**Redshifts.** The forest snapshots span **z ≈ 5.4 → 2.2 in Δz = 0.2 steps**
(the science range used by the PRIYA flux-power emulator). Snapshot→redshift is
read from each file's own header (indices are *not* a fixed z map and have gaps),
so the loader never assumes a redshift from a directory index.

**Lyman-α optical-depth (`tau`) product.** Per snapshot, `fake_spectra`'s gridded
output `output/SPECTRA_<NNN>/lya_forest_spectra_grid_480.hdf5` holds HI Lyα
(`tau/H/1/1215`, float32) on a **regular 480×480 transverse grid × 3 line-of-sight
axes (x, y, z) = 691,200 skewers**. Transverse spacing = box/480 = 250 ckpc/h; the
LOS has `nbins` velocity pixels (z-dependent, ≈1100–1750 — more pixels at higher z) at ≈10 km/s. `priya_loader`
returns one axis as a `(480, 480, nbins)` cube, unmodified.

**Initial-condition density.** Each simulation's `ICS/<box>_<Ngrid>_99/` is an
MP-GenIC `bigfile` of gas (type 0) and DM (type 1) particles (comoving kpc/h
positions). `load_ic_density` streams one type and CIC-paints it to an `nmesh³`
real-space overdensity `δ = ρ/⟨ρ⟩−1` (float32) at `z_init≈99`.

> **Partial staging.** A given machine often holds only part of the suite
> mid-transfer (e.g. only the high-z snapshots, or some simulations with no `tau`
> files yet). All discovery helpers degrade gracefully, and you can inventory a
> tree directly: `paths.discover_simulations(root)` +
> `paths.find_spectra_files(sim_dir)` report exactly what is present.

## Units

MP-Gadget internal units (comoving **kpc/h**, **10¹⁰ M☉/h**, **km/s** with the
Gadget √a velocity convention) and all conversions live in
[`priya_loader/units.py`](priya_loader/units.py), with a dedicated physics-unit
test suite — the easiest place to get a subtle factor wrong.

## Development

```bash
pip install -e ".[dev]"
pytest                      # hermetic: no real data or network required
```

The default `pytest` run uses synthetic fixtures only, so it runs anywhere. Two
opt-in gates exercise genuine data and **auto-skip** otherwise:

```bash
# τ / flux on the real grid_480 files (tests marked `realdata`):
PRIYA_DATA_ROOT=/path/to/priya/emu_full pytest -m realdata

# real MP-GenIC IC sign-off, pointed at ONE staged IC bigfile directory
# (low-res = 512³ `120_512_99`; hi-res = `120_3072_99`, resolution test only):
PRIYA_REAL_IC=/path/to/priya/emu_full/<sim>/ICS/120_512_99 pytest tests/test_real_ic.py
```

The built-in dependency-free CIC reference always runs; the Pylians/nbodykit
CIC cross-checks additionally need those libraries (Pylians builds only in a
`numpy<2` env). See `tests/test_real_ic.py` for the full gate documentation.

## Provenance

Every non-obvious convention this package encodes — the MP-Gadget unit constants,
the `Ap` amplitude pivot, the simulation folder-name format, and the
`fake_spectra` tau layout / axis ordering — is traced to its upstream source
(repo, file:line, pinned commit) in [PROVENANCE.md](PROVENANCE.md).

## License

MIT — see [LICENSE](LICENSE).
