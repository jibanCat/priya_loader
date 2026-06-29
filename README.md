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

The headline object (coming in a later release) loops over every
simulation/redshift and yields, per snapshot:

```python
[ SimParams, redshift, ic_density_3d, tau_3d ]
```

so a downstream pipeline (e.g. a JAX bias-estimation code) can `np.load` and go.

> **Status.** This version ships the building blocks (`units`, `params`,
> `runconfig`, `paths`) **plus the Lyman-α `tau` loader** (`load_tau_grid`). The
> IC density loader and the `PriyaDataset` orchestrator land in subsequent releases.

## Install

Lightweight (flux fields + exported `.npz`; runs on a NERSC login node, no MPI):

```bash
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

`nbodykit` (an optional alternative meshing backend) is **not** a pip extra —
install it manually from conda-forge if you want it.

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
**anisotropic**: 250 ckpc/h transverse (`box/480`) vs ~70 ckpc/h along the LOS
(`box/nbins`), and the LOS pixel is ≈10 km/s (`g.meta["dv_kms"]`). Each axis cube
is ~1.5 GB (`230 400 × nbins` float32); the loader reads **one axis at a time** so
it fits a NERSC login node — all three axes would be ~4.6 GB.

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
Each simulation is one point in a 9-dimensional Latin hypercube:

| group | parameter | symbol | range (emu_full) |
|-------|-----------|--------|------------------|
| cosmology | `ns` | n_s | 0.80 – 1.04 |
| cosmology | `Ap` (folder) / `scalar_amp` | A_P(k=0.78) | 1.2 – 2.6 ×10⁻⁹ |
| cosmology | `hubble` | h | 0.65 – 0.75 |
| cosmology | `omega_m_h2` | Ω_m h² | 0.140 – 0.146 |
| HeII reion. | `here_i`, `here_f`, `alpha_q` | z_i, z_f, α_q | 3.5–4.5, 2.2–3.2, 1.3–3.0 |
| HI reion. | `hireionz` | z_HI | 6.5 – 8.0 |
| feedback | `bhfeedback` | ε_BH | 0.031 – 0.069 |

**Redshifts.** The forest snapshots span **z ≈ 5.4 → 2.0 in Δz = 0.2 steps**
(the science range used by the PRIYA flux-power emulator). Snapshot→redshift is
read from each file's own header (indices are *not* a fixed z map and have gaps),
so the loader never assumes a redshift from a directory index.

**Lyman-α optical-depth (`tau`) product.** Per snapshot, `fake_spectra`'s gridded
output `output/SPECTRA_<NNN>/lya_forest_spectra_grid_480.hdf5` holds HI Lyα
(`tau/H/1/1215`, float32) on a **regular 480×480 transverse grid × 3 line-of-sight
axes (x, y, z) = 691,200 skewers**. Transverse spacing = box/480 = 250 ckpc/h; the
LOS has `nbins` velocity pixels (z-dependent, ≈1570–1750) at ≈10 km/s. `priya_loader`
returns one axis as a `(480, 480, nbins)` cube, unmodified.

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

Tests use synthetic fixtures only, so they run anywhere. Tests marked `realdata`
(which load genuine multi-GB `grid_480` files) are **auto-skipped** unless you
point at a staged tree:

```bash
PRIYA_DATA_ROOT=/path/to/priya/emu_full pytest -m realdata
```

## Provenance

Every non-obvious convention this package encodes — the MP-Gadget unit constants,
the `Ap` amplitude pivot, the simulation folder-name format, and the
`fake_spectra` tau layout / axis ordering — is traced to its upstream source
(repo, file:line, pinned commit) in [PROVENANCE.md](PROVENANCE.md).

## License

MIT — see [LICENSE](LICENSE).
