# Provenance — sources for the conventions this package relies on

Every non-obvious factual claim in the code is traced here to an upstream source
(repository, file:line, pinned commit) or to an explicit empirical verification
against the real data. Pinned commits below:

| upstream | commit / tag | as of |
|---|---|---|
| [MP-Gadget](https://github.com/MP-Gadget/MP-Gadget) | `471711f80daa262f68a2d6873bdc3fe471a631da` (master; no tags) | 2026-06-19 |
| [MP-Gadget](https://github.com/MP-Gadget/MP-Gadget) (IC velocity convention re-verification) | `12f2c82a61bb7fa83685c8eb4b459b83710e0580` (master; no tags) | 2026-07-13 |
| [fake_spectra](https://github.com/sbird/fake_spectra) | `93d0e509c13363b2e5c34db294661d658de2f81b` (tag `v2.2.3`; installed) | — |
| [lya_emulator](https://github.com/sbird/lya_emulator) | `27dac4f6c89b9126e141ed58316aff613d311c4f` (master) | 2025-10-24 |
| [SimulationRunner](https://github.com/sbird/SimulationRunner) | `5adf4fe25ea7c376394327b47da481f663466788` (master) | 2025-09-22 |
| [bigfile](https://github.com/rainwoodman/bigfile) | `eeaaa75acf85d5eebd6196f9bb55d54db513fe18` (master; installed `0.1.52`) | 2026-05-02 |
| [Pylians3](https://github.com/franciscovillaescusa/Pylians3) | `5bfaf0006a80a2aa2f0f33f309d68b7ac3172b2d` (master; installed `Pylians 0.12`) | 2026-02-23 |
| [nbodykit](https://github.com/bccp/nbodykit) | `c29f379b3b58b52d3294419b8f85f623d672f5a7` (master; not installed — numpy<2 only) | 2025-11-07 |

Line numbers are from the pinned commits.

**Trace any citation yourself.** Every citation below was verified to resolve to
the claimed code at its pinned commit. Open the raw file or the line-anchored
view (line numbers may have drifted ±1 between commits; search the token if so):

```
raw:  https://raw.githubusercontent.com/<owner>/<repo>/<commit>/<path>
view: https://github.com/<owner>/<repo>/blob/<commit>/<path>#L<line>
# e.g. https://github.com/MP-Gadget/MP-Gadget/blob/471711f8.../genic/params.c#L65
#      https://github.com/sbird/fake_spectra/blob/93d0e509.../fake_spectra/griddedspectra.py#L42
```

## `units.py` — MP-Gadget internal units (@ 471711f8)

| claim | source |
|---|---|
| `UNIT_LENGTH_IN_CM = 3.085678e21` (1 kpc/h) | `genic/params.c:65` (default `CM_PER_MPC/1000`); reader `libgadget/petaio.c:501` (literal) |
| `UNIT_MASS_IN_G = 1.989e43` (1e10 Msun/h) | `genic/params.c:66`; `libgadget/petaio.c:502` |
| `UNIT_VELOCITY_IN_CM_S = 1e5` (1 km/s) | `genic/params.c:64`; `libgadget/petaio.c:500` |
| `v_pec = u_stored * sqrt(a)` | `libgenic/zeldovich.c:201` `vel_prefac /= sqrt(GenicConfig.TimeIC)`. **Caveat:** skipped when header `UsePeculiarVelocity == 1` (then Velocity is already peculiar km/s). |
| `RHO_CRIT_1E10_MSUN_H = 27.7537` | Derived: `3 H0^2/(8 pi G)`. MP-Gadget computes it at `libgadget/cosmology.c:21` (with `G=6.672e-8`, `H=3.2407789e-18`, `physconst.h`) → `27.755`; agrees with the textbook `27.7537` to ~4 sig figs (rel. diff ~5e-5). |
| particle mass `Omega * RHO_CRIT * box^3 / Ngrid^3` | `libgenic/save.c:106` (CDM), `:96` (gas) |

## `params.py` / `paths.py` / `runconfig.py` — naming & parameters

| claim | source |
|---|---|
| Folder name = `<name><value>` concatenation, `"%.3g"` | `lyaemu/coarse_grid.py:106-118` `build_dirname` @ 27dac4f |
| Parameter order `ns, Ap, herei, heref, alphaq, hub, omegamh2, hireionz, bhfeedback` | `lyaemu/coarse_grid.py:41` `param_names` @ 27dac4f |
| `Ap = scalar_amp * ((2π/8)/0.05)^(ns−1)` (Ap = A at the 8 Mpc scale, `k ≈ 0.785 Mpc⁻¹`; scalar_amp = A_s at `k=0.05 Mpc⁻¹`) | `lyaemu/coarse_grid.py:154-157, 259-265` @ 27dac4f (the literal "0.78" in the name is the code's rounding of `2π/8`) |
| `SimulationICs.json` `box`/`npart` unreliable | `SimulationRunner simulationics.py:267-285` @ 5adf4fe — `txt_description()` snapshots `__dict__` once at IC-build and is not rewritten when GenIC is regenerated. (`box`/`npart` are required args, *not* defaults; the `15`/`192` values came from the generating script.) Empirically: 30/60 emu_full JSONs read `box=15, npart=192` while the run is 120/1536. |
| IC dir name `<box>_<Ngrid>_<z_init>` (e.g. `120_1536_99`) | GenIC `FileBase`/`OutputDir`, built at `SimulationRunner/simulationics.py:210-211` @ 5adf4fe (the template `mpgenic.ini` only holds the default `FileBase = IC`) |
| Production box/Ngrid from `mpgadget.param`/`_genic_params.ini` | MP-Gadget / MP-GenIC parameter files |
| `omega_lambda = 1 − omega0` (flat; radiation handled separately) | MP-Gadget `libgadget/cosmology.c` (`OmegaLambda = 1 − Omega0 − OmegaR`, `OmegaR` from `OmegaG`/`OmegaNu`) @ 471711f8; matches GenIC `OmegaLambda` |
| PRIYA suite: box 120 Mpc/h, low-res 1536³ + hi-res 3072³, 9-param Latin hypercube, fixed photo-heating amplitude | PRIYA: Bird et al. 2023, [arXiv:2306.05471](https://arxiv.org/abs/2306.05471) (JCAP 10 (2023) 037). On-disk `_genic_params.ini` is the per-sim empirical stamp. |

## `ic.py` / `mesh.py` — IC particle reading (@ MP-Gadget 471711f8)

| claim | source |
|---|---|
| particle type `0 = gas`, `1 = DM` (bigfile block prefix) | MP-Gadget / MP-GenIC particle-type convention (`libgenic/save.c`) |
| `Position` is comoving kpc/h in `[0, BoxSize)` | `genic/params.c:65` (UnitLength = kpc/h); positions written `libgenic/save.c:73` |
| `Position` = Lagrangian grid `q` + **1LPT Zel'dovich** displacement `Ψ` (no 2LPT) | `libgenic/zeldovich.c:243-244` (`Pos += Disp`); `q` from `idgen_create_pos_from_index` `zeldovich.c:85,98`; `Ψ(k)=ik/k²·δ(k)` `zeldovich.c:293,308` @ 471711f8 |
| `icdensity` Lagrangian decode `id = i·Ng² + j·Ng + k + 1` (i=x slowest, k=z fastest; C-order, 1-indexed) ⇒ within-type index `(ID−1) mod ngrid³` | `libgenic/zeldovich.c` `idgen_create_id_from_index` (~L48, `id = i*Ngrid*Ngrid + j*Ngrid + k + 1`) @ 471711f8. Per-type `FirstID = m·ngrid³`: `genic/main.c:192,198` (DM low / gas high block; see Notes). Used by `_load_icdensity_grid` (`ic.py:204-267`). |
| native `ICDensity` block = **linear density contrast `δ₁`** on the Lagrangian grid (= −∇·Ψ, 1st order), ~1-cell smoothed | `libgenic/save.c:72`; filled by the "Density" transfer `zeldovich.c:182,284` = white noise × `DeltaSpec(k)` (normalized linear δ, `power.c:52-65`) @ 471711f8. *(Read by `load_ic_density(..., field="icdensity")`.)* |
| CIC overdensity `δ=ρ/⟨ρ⟩−1`; raw (uncompensated) window `W(k)=∏ sinc²(k_iΔ/2)`, `Δ=box/nmesh` | Cloud-in-cell: Hockney & Eastwood, *Computer Simulation Using Particles* (1981) §5-3. Window form: Jing 2005, arXiv:astro-ph/0409240 eqs (3),(7). We ship **raw** (no deconvolution); equivalence of the built-in painter to Pylians/nbodykit is **empirical** (see external-deps section). |
| IC linear P(k)/transfer generated by **CLASS** (the `classy` Python binding), not CAMB | `SimulationRunner/simulationics.py::cambfile()` (`from classy import Class`, `.compute()`, lines 12-13,169-187; `camb_git = classy.__version__` line 444) @ 5adf4fe. The staged sims have `camb_git: 0.2.9` (the older `classylss` binding — also CLASS). `camb_linear/`/`camb_git` are legacy CAMB labels. |

## `ic.py` / `units.py` — IC velocity convention (@ MP-Gadget 12f2c82a)

Re-verified against a later MP-Gadget master than the `471711f8` pin above (line
numbers below are current as of `12f2c82a`); used by `load_ic_particles`'
`velocity="peculiar_kms"` default and `units.ic_velocity_to_peculiar_kms`.

| claim | source |
|---|---|
| `UsePeculiarVelocity` declared `OPTIONAL, 1` (default **1**) | `genic/params.c:52` |
| `vel_prefac = a·H(a)`; divided by `sqrt(a)` **only** when the flag is 0 — i.e. flag=1 (PRIYA) means the stored `Velocity` block is already peculiar km/s, no `sqrt(a)` rescale | `libgenic/zeldovich.c:195-202` |
| The `UsePeculiarVelocity` flag is written into the IC `Header` attrs (so the loader reads it back rather than assuming) | `libgenic/save.c:128` |
| `ScaleDepVelocity` defaults to `DifferentTransferFunctions` (=1 in PRIYA's `_genic_params.ini`) ⇒ velocities are drawn from per-species CLASS `VelX/Y/Z` transfer functions, not a rescaled Zel'dovich displacement — gas and DM velocities differ | `genic/params.c:47,141-144` |

PRIYA's `_genic_params.ini` never sets `UsePeculiarVelocity`, so it takes the
GenIC default (1); applying the Gadget-2 `sqrt(a)` to an already-peculiar block
is a **10× error at z=99** (`sqrt(100) = 10`).

## `tau.py` — fake_spectra gridded product (@ v2.2.3, 93d0e509)

| claim | source |
|---|---|
| tau key `tau/<elem>/<ion>/<lambda>` = `H/1/1215` (HI Lyα) | `spectra.py:332-350` (`_save_multihash`); line value `int(1215.67)=1215` |
| 3-axis grid: `Nlos = 3·ngrid²`, axis 1/2/3 blocks contiguous | `griddedspectra.py:34-60` (3-axis mode) |
| **`CUBE_AXES` mapping** axis1→(y,z,x), axis2→(x,z,y), axis3→(x,y,z), lower-numbered transverse coord = slow/outer index | `griddedspectra.py:42-60`: `grid_id = [0,nn,mm]/[nn,0,mm]/[nn,mm,0]`, `for nn ... for mm` (C-order). **Also verified empirically** against `spectra/cofm` for all 3 axes (held LOS coord = 0; `box/ngrid` = 250 ckpc/h spacing). |
| tau is **redshift-space** (LOS = velocity axis) | `absorption.cpp:234` `vel = velfac*pos1 + pvel` (Hubble + peculiar), binned in velocity |
| transverse spacing `dx = box/ngrid` | `griddedspectra.py:57` `dx = self.box/(1.*nspec)` (250 ckpc/h is the PRIYA value, not a constant) |
| pixel `dv ≈ res` (~10 km/s) | `spectra.py` `dvbin = vmax/int(vmax/res)` (≈ `res` up to `int()`; library asserts to `rtol=1e-2`) |
| `F = exp(-tau)` | `fluxstatistics.py` (mean-flux / `_rescale_mean_flux`) |
| Header attrs (box ckpc/h, hubble, omegam/b/l, Hz, nbins, redshift) written by fake_spectra | `spectra.py` `save_file` Header writes |

## `units.py` — cosmology helpers (growth, Hubble, velocity)

| claim | source |
|---|---|
| `growth_factor` `D(a) ∝ H(a) ∫₀ᵃ da′/(a′H(a′))³`, normalized `D(0)=1` (the `5Ωm/2` prefactor cancels) | Eisenstein 1997, [arXiv:astro-ph/9709054](https://arxiv.org/abs/astro-ph/9709054) (analytic growth, flat ΛCDM); Peebles 1980 *LSS* §10. **Radiation neglected** (small error in `D(99)`). |
| `growth_rate` `f(z) ≈ Ωm(z)^0.55` (growth index γ=0.55) | Linder 2005, [arXiv:astro-ph/0507263](https://arxiv.org/abs/astro-ph/0507263) (PRD 72 043529). **Distinct** from GenIC's IC-velocity `F_Omega ≈ Ω(a)^0.6` (EdS approx, `libgenic/zeldovich.c` ~L205 @ 471711f8). |
| `hubble_z = 100 h √(Ωm(1+z)³ + ΩΛ)` (flat ΛCDM, radiation neglected) | Friedmann eq. (Dodelson, *Modern Cosmology* §2). **Byte-for-byte** the product's stored `Hz`: fake_spectra `spectra.py` `self.Hz = 100*hubble*sqrt(OmegaM/atime³+OmegaLambda)` @ 93d0e509 (v2.2.3). |
| `velfac = H(z)/(h(1+z))` (comoving Mpc/h → km/s) | Same convention as fake_spectra `vel = velfac·pos1 + pvel` `absorption.cpp:234` + `spectra.py` `velfac` @ 93d0e509. |

## External meshing & IO libraries (the tests cross-check the built-in CIC against these)

The numerical *equivalence* of `priya_loader.mesh.cic_paint` to these libraries is an
**empirical** guarantee — codified by the shipped `np.testing.assert_allclose` tests
(`tests/test_mesh.py`, `tests/test_real_ic.py`) — not a derived identity. The
dependency-free per-particle reference (`tests/_explicit_cic`, Hockney & Eastwood
1981 §5-3) is the primary anchor that runs without any optional dependency.

| claim | source |
|---|---|
| `bigfile` `.size` = aggregate record count over `Nfile` (not bytes) — used by `tests/test_real_ic.py` to read `npart` | bigfile `bigfile/pyxbigfile.pyx:313-314` (`property size: return self.bb.size`) @ eeaaa75a (installed 0.1.52) |
| Pylians CIC: `MAS_library.MA(pos_f32, grid, BoxSize, "CIC")` paints a **raw mass grid** (so `pyl/pyl.mean()−1 = δ`); positions float32 in `[0,BoxSize]` | Pylians3 `library/MAS_library/MAS_library.pyx` `MA(...)` (~L56) → `CIC(...)` (~L74), `inv_cell_size = dims/BoxSize` (~L284), in-place `number[...] += d0*d1*d2` (~L292-299) @ 5bfaf000 (installed Pylians 0.12) |
| nbodykit CIC: `ArrayCatalog({"Position":pos}).to_mesh(Nmesh, BoxSize, resampler="cic", compensated=False, interlaced=False)` returns **unit-mean `1+δ`** (so `compute("real")−1 = δ`) | nbodykit `base/catalog.py::to_mesh` (defaults `interlaced=False, compensated=False, resampler='cic'`, ~L775); `source/mesh/catalog.py::to_real_field` `nbar=W/∏Nmesh` (~L328) → normalize (~L356-358); `CompensateCIC` = the window we omit @ c29f379b |

## Notes / discrepancies recorded
- **ID block ordering** (relevant to the `field="icdensity"` route): in current
  MP-Gadget master, DM (type 1) gets the low ID block and gas (type 0) the high
  block (`genic/main.c:192,198`), the *opposite* of an older GenIC. The loader is
  invariant to this: within a type the row-major index is `(ID-1) mod ngrid³`
  (per-type `FirstID` is a multiple of `ngrid³`), and a completeness check
  (`ID.max()-ID.min()+1 == npart`) rejects partial blocks.
