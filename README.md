# priya_loader

ML-friendly access to the **PRIYA** Lyman-α simulation suite — a grid of
MP-Gadget cosmological hydrodynamic simulations. `priya_loader` turns each
simulation into clean 3-D numpy arrays so you can treat the suite as a dataset:

- the **initial-condition (IC) density field** (from the MP-GenIC `bigfile`
  ICs, painted onto a mesh), and
- the **Lyman-α flux field** `δ_F` at each output redshift (from the
  pre-computed gridded `fake_spectra` optical-depth files, read with `h5py`).

The headline object (coming in a later release) loops over every
simulation/redshift and yields, per snapshot:

```python
[ SimParams, redshift, ic_density_3d, flux_3d ]
```

so a downstream pipeline (e.g. a JAX bias-estimation code) can `np.load` and go.

> **Status — foundation release.** This version ships the building blocks
> (`units`, `params`, `runconfig`, `paths`). The flux loader, IC loader, and the
> `PriyaDataset` orchestrator land in subsequent releases.

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
    flux_files = paths.find_spectra_files(sim.directory)   # [(snap_index, path), ...]
    print(p.name, p.fidelity, f"box={p.box} Mpc/h", f"npart={p.npart}",
          f"z=[{p.z_end},{p.z_init}]", f"flux_snaps={len(flux_files)}")
    feature_vector = p.as_vector(["ns", "scalar_amp", "hubble", "omega0", "bhfeedback"])
```

### Why a `runconfig` module?

The per-simulation `SimulationICs.json` carries **stale template `box`/`npart`**
for a large fraction of the suite (it reads `box=15, npart=192` even for
production 120 Mpc/h, 1536³ runs). `priya_loader` therefore reads the box and
particle grid from the authoritative run files (`mpgadget.param`,
`_genic_params.ini`) and uses the JSON only for cosmology/astrophysics. This is
handled for you in `SimParams.from_dir`.

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

Tests use synthetic fixtures only, so they run anywhere.

## License

MIT — see [LICENSE](LICENSE).
