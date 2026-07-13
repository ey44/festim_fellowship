# FESTIM Model Plan: Tritium Permeation in a He-Purge Heat Exchanger

## 1. Objective

Estimate the tritium flux from a He purge stream (primary side) through the heat
exchanger wall into the secondary coolant, as a function of:

- He mass flow rate (baseline: 1 kg/s)
- Tritium partial pressure in the He stream (baseline: 10 Pa, as T2)
- He temperature, which drops from `T_hot_inlet = 600 C` to `T_hot_outlet = 300 C`
  along the length of the exchanger
- Secondary side: water/steam, counterflow, `T_cold_inlet = 290 C` to
  `T_cold_outlet = 580 C`

The deliverable is a tritium permeation rate (mol/s, or Ci/day) into the secondary
coolant, plus its sensitivity to flow rate and partial pressure — eventually as an
artifact others can re-run with different operating points.

## 2. Placeholder system definition

Since exact geometry/material weren't specified, the plan uses placeholders that can
be swapped later without changing the model structure:

- **Wall material:** Inconel 617 (typical HX structural alloy for He loops). Use
  literature `D_0`, `E_D` for H/T diffusivity, and Sieverts solubility constants
  `S_0`, `E_S`. Source from `material_library.py` if entries exist, otherwise add
  an Inconel 617 entry there following the same pattern as the existing materials.
- **Geometry:** tube bank, OD/ID giving a wall thickness `t_wall ~ 2-3 mm`.
  Tube count `N` and length `L` are NOT free parameters — they come from the
  thermal sizing in Phase 0 (Section 4), which fixes the total heat-transfer
  area `A`. The permeation area is the same `A`, so the tritium answer is
  self-consistent with the thermal duty instead of scaling with an arbitrary
  length.
- **Tritium species:** treat as T2 in the He stream (consistent with the 10 Pa
  spec), Sieverts law on the metal surface (`c_s = K_S * sqrt(P_T2)`).
- **Coolant side BC:** zero-concentration sink (`FixedConcentrationBC = 0`).
  For a water/steam secondary this is more than a conservative placeholder:
  tritium reaching the steam side converts to HTO and is carried away, so the
  back-pressure is effectively zero. A finite mass-transfer resistance can still
  be added later the same way `transport.py` does for the permeator (K_eff on
  that face), but the zero-sink case is close to physical here.

## 3. Why a staged approach

A single fully-coupled 2D FESTIM model (axial flow direction + through-wall
diffusion, with conjugate heat transfer) is the "correct" final answer, but it's
slow to build and debug, and the He-side boundary condition (Sieverts + gas-film
resistance, both temperature- and flow-dependent) is easiest to get right in
isolation. So the plan builds up in four phases, each one a checkpoint that can be
validated before adding complexity.

## 4. Phase 0 — HX thermal sizing (geometry from duty, U, LMTD)

**Goal:** derive the heat-transfer area `A` (and hence `N` tubes x length `L`)
from the thermal duty, so the permeation area is physically determined rather
than assumed.

- **Duty (fixed by the He side):**
  `Q_dot = mdot_He * cp_He * (600 - 300) = 1 kg/s * 5193 J/kg.K * 300 K ~ 1.56 MW`.
- **LMTD (no assumption needed):** counterflow with steam 290 -> 580 C gives
  `dT_hot_end = 600 - 580 = 20 C`, `dT_cold_end = 300 - 290 = 10 C`, so
  `LMTD = (20 - 10)/ln(20/10) = 14.4 C`. This is a tight-pinch design (10 C at
  the cold end).
- **Capacity-rate check:** `C_hot = 5.19 kW/K`, `C_cold = Q_dot/290 K = 5.37 kW/K`
  — ratio 1.03, i.e. nearly balanced streams. Consequence: the counterflow
  temperature profiles are very close to linear in `z` and the local `dT` is
  ~10-20 C everywhere. The "linear T(z)" used in Phase 3 is therefore a good
  approximation here, not just a placeholder.
- **Single-phase caveat:** the straight-LMTD treatment is valid only if the
  secondary is single-phase across the exchanger (supercritical pressure, or
  already superheated steam at 290 C). If the secondary boils (subcritical,
  290 C ~ liquid), the exchanger must be split into economizer/evaporator/
  superheater zones with separate LMTDs and an internal pinch-point check — the
  single-LMTD area would be wrong. Confirm secondary pressure before trusting
  the sizing.
- **Overall U:** `1/U = 1/h_He + t_wall/k_wall + 1/h_sec` (referenced to one
  area; add the cylindrical correction if OD/ID matters). `h_He` from a Nusselt
  correlation (Dittus-Boelter/Gnielinski) — add a `nusselt_number()` to
  `transport.py` mirroring the existing `sherwood_number()` (same Re machinery).
  `h_sec` for water/supercritical steam is large (several kW/m2.K), so U is
  He-side dominated; expect `U ~ 300-600 W/m2.K`.
- **Area:** `A = Q_dot / (U * LMTD)`. E.g. `U = 400 W/m2.K` -> `A ~ 270 m2`.
  Choose tube ID/OD and tube count `N` from a He-side velocity/Re target (Re
  should land in the same regime the `sherwood_number()`/`nusselt_number()`
  correlations assume), then `L = A / (N * pi * d)`.
- **Sensitivity to report:** `Q_T` (total permeation) scales ~1/U via the area,
  so carry U as an explicit uncertainty band (e.g. 300-600 W/m2.K) through to
  the final answer, alongside the P_T2 and flow-rate sweeps.
- Implemented in `hx_thermal_sizing.py`; outputs `A`, `N`, `L`, `U`, `T(z)` and
  `T_wall(z)` for the downstream phases.

## 5. Phase 1 — 1D through-wall steady-state model (single operating point)

**Goal:** get a working FESTIM model that reproduces a single local tritium flux
through the wall, for one (T, P_T2, flow rate) combination.

- `F.Mesh1D` across the wall thickness (`0` to `t_wall`), material = Inconel 617.
- Uniform temperature `my_model.temperature = T` (steady, no heat solve yet).
- He-side surface BC: start with `F.SievertsBC(pressure=P_T2, S_0=..., E_S=...)`.
  As a second step, replace this with a custom `F.ParticleFluxBC` that uses the
  combined feed-side resistance `K_eff_f(T, P, Q, ...)` already implemented in
  `transport.py` (gas-film + surface kinetics), so the He flow rate enters the
  model exactly as it does in the permeator. This is the mechanism by which "vary
  the flow rate" actually changes the result — at fixed T and P_T2, a higher flow
  rate increases `k_m` and therefore `K_eff_f`, raising the effective surface
  concentration toward the Sieverts limit.
- Coolant-side surface BC: `F.FixedConcentrationBC(value=0)`.
- Export: `F.SurfaceFlux` on the coolant-side surface — this is the local tritium
  flux `J_perm(T, P_T2, Q_He)` in mol/m^2/s.
- **Validation:** for the pure-SievertsBC case with a zero-concentration sink, the
  steady-state analytical solution is `J = D(T) * K_S(T) * sqrt(P_T2) / t_wall`.
  Compare FESTIM's `SurfaceFlux` output against this closed form — this is the
  first verification check in Section 7.

## 6. Phase 2 — Flux map over the operating envelope

**Goal:** turn Phase 1 into a function `J_perm(T, P_T2, Q_He)` evaluated over the
ranges relevant to the HX.

- Discretize the axial temperature range into N stations between `T_hot_inlet`
  (600 C) and `T_hot_outlet` (300 C) — e.g. 7-10 stations. Use the **wall**
  temperature `T_wall(z)` from Phase 0, not the gas temperature: permeation is
  set by the metal's T. With this tight-pinch design the heat flux is modest
  (`q'' = U * dT_local ~ 4-9 kW/m2`), so `T_wall ~ T_He - q''/h_He` is only a
  few C below the gas — small, but free to include since Phase 0 computes it.
- For each station, run the Phase 1 model with that station's T, the chosen
  `P_T2`, and `Q_He`, recording `J_perm`.
- Repeat across a grid of `P_T2` (e.g. 1-100 Pa) and `Q_He` (e.g. 0.1-2 kg/s) to
  build the sensitivity surfaces requested ("vary flow rate and partial
  pressure").
- Output: a small dataset/table (`flux_map.csv` or similar) of
  `J_perm(T, P_T2, Q_He)`, plus diagnostic quantities (K_eff_f, regime indicator —
  diffusion-limited vs surface-limited, reusing `Lambda`/`lambda_regime` from
  `transport.py`/`permeator_v3.py`) so we know which resistance dominates across
  the operating envelope.

This phase is just many independent Phase-1 runs — no new FESTIM physics, just a
driver script (`run_flux_map.py`) that loops over (T, P_T2, Q_He) and calls the
Phase-1 model.

## 7. Phase 3 — Axial integration to total permeation rate

**Goal:** combine the flux map with the actual axial temperature profile and HX
geometry to get a single number: total tritium permeation rate `Q_T` [mol/s] into
the secondary coolant.

- Take `T(z)` (and `T_wall(z)`) directly from the Phase 0 counterflow energy
  balance. Because the capacity rates are nearly balanced (C ratio 1.03), the
  exact counterflow profile is close to linear — use the exact one since it's
  already computed, and note the linear approximation is justified here.
- Optionally track tritium depletion in the He stream along `z`, mirroring the
  ODE structure in `permeator_v3.py`:
  `dF_T2/dz = -J_perm(T(z), P_T2(z), Q_He) * (wetted perimeter)`,
  so `P_T2(z)` decreases as tritium permeates out. For a first pass this can be
  neglected (assume `P_T2` ~ constant) if the fractional tritium loss is small —
  flag this as an assumption to check.
- Integrate: `Q_T = integral over z of J_perm(T_wall(z), P_T2(z), Q_He) *
  (N * perimeter) dz`, with `N` and perimeter from Phase 0, using the flux map
  from Phase 2 (interpolated) rather than re-running FESTIM at every z. Total
  integration area must equal the Phase 0 `A` — cheap consistency check.
- Output: `Q_T` [mol/s] and equivalent activity [Ci/day or Bq/s], reported for the
  baseline case and across the flow-rate/partial-pressure sensitivity grid.

## 8. Phase 4 — 2D coupled model (validation / final model)

**Goal:** build a single 2D FESTIM model (axial x through-wall) that solves heat
transfer and hydrogen transport together, to check Phases 1-3's "stitched 1D"
approximation.

- 2D mesh: axial extent `L`, through-wall extent `t_wall`.
- Wall temperature field `T(z, y)` PRESCRIBED from the Phase 0 thermal
  solution (He-side and steam-side wall-surface profiles, linear through the
  thin wall) rather than a coupled `F.HeatTransferProblem`. For a 2.5 mm wall
  with known film coefficients this is essentially exact and far more robust;
  the coupled heat solve remains a possible later refinement (swap the
  `temperature` argument).
- Hydrogen transport on the mesh with that temperature field:
  - He-side surface BC: same Sieverts/K_eff_f BC as Phase 1, now spatially varying
    via the solved `T(x,z)`
  - Coolant-side: `FixedConcentrationBC = 0`
- Export: `SurfaceFlux` integrated along the coolant-side boundary -> directly
  gives `Q_T`, to compare against Phase 3's integrated result.
- This is the most expensive phase computationally and is the natural place to
  stop unless the 1D approximation in Phase 3 turns out to be inadequate (e.g. if
  axial conduction in the wall or strong nonlinearity in `J_perm(T)` makes the
  "local 1D slice" assumption break down).

## 9. Verification & validation checklist

- Phase 0: energy-balance closure (He-side duty = secondary-side duty within
  cp uncertainty); cross-check the LMTD area against an effectiveness-NTU
  calculation (`epsilon = 300/310 = 0.97`, near-balanced counterflow — note
  this high effectiveness is why the area is large); sanity-check `h_He` and U
  against typical gas-HX values.
- Phase 1: FESTIM `SurfaceFlux` vs. analytical `J = D*K_S*sqrt(P_T2)/t_wall`
  (zero-sink, pure Sieverts case), at 2-3 temperatures. Also a mesh-convergence
  check (refine `F.Mesh1D` and confirm `J_perm` converges).
- Phase 2: spot-check that `K_eff_f` and `Lambda` values reproduce the same
  regime classification (`Lambda_regime`) as the existing permeator model at
  comparable T/P, since both use the same `transport.py` correlations.
- Phase 3: check sensitivity of `Q_T` to the number of axial integration stations
  (convergence), to U (300-600 W/m2.K band -> area -> Q_T), and confirm the
  integrated area equals the Phase 0 `A`.
- Phase 4: compare integrated `Q_T` against Phase 3; investigate any discrepancy
  in terms of axial wall conduction or temperature-dependence of `J_perm`.

## 10. Proposed file structure

```
My Own Tutorials/Heat Exchanger Model/
  HX_FESTIM_plan.md          <- this document
  hx_thermal_sizing.py        # Phase 0: duty, LMTD, U, A, N, L, T(z), T_wall(z)
  hx_1d_model.py              # Phase 1: single-point 1D FESTIM model
  run_flux_map.py             # Phase 2: sweep (T, P_T2, Q_He) -> flux_map.csv
  flux_map.csv                # Phase 2 output
  axial_integration.py        # Phase 3: T(z), P_T2(z), Q_T
  hx_2d_model.py               # Phase 4: coupled 2D model
  material_library.py          # shared (link/copy from permeator_model copy/, add Inconel 617)
  transport.py                  # shared (reuse K_eff_f, Lambda from permeator_model copy/)
```

## 11. Open assumptions to revisit

- Wall material and `t_wall` are placeholders (Inconel 617, ~2-3 mm) — update
  once actual HX design values are available. Geometry (`A`, `N`, `L`) is now
  derived in Phase 0, but inherits the U uncertainty below.
- RESOLVED — wall curvature: the wall is thick relative to the bore
  (2.5 mm on r_i = 5 mm), so the flat-slab approximation underestimates
  permeation by ~19%. Phases 1-3 now use the exact annular solution via an
  effective thickness `t_eff = r_i ln(r_o/r_i)` (geometry='cylindrical',
  default); Phase 0 likewise uses the cylindrical wall resistance. The 2D
  FESTIM model remains a Cartesian slab (its job is to test the stitched-1D
  approximation, compared like-for-like against Phase 3 in slab mode); a
  cylindrical 1D FESTIM check (`compute_flux_festim_cylindrical`) verifies
  `t_eff` directly, with manual flux extraction since FESTIM's SurfaceFlux
  export is Cartesian-only.
- **Secondary-side pressure / phase:** the single-LMTD sizing assumes the
  water/steam stream is single-phase from 290 to 580 C. If subcritical with
  boiling, redo Phase 0 with zoned LMTDs and a pinch-point check.
- **Overall U (300-600 W/m2.K assumed range):** the largest geometry
  uncertainty; `A ~ 1/U` so it propagates linearly into `Q_T`. Tighten with the
  actual tube layout and `h_He` correlation once known.
- Tritium speciation in He assumed to be T2 (matches the 10 Pa spec); if HT is
  more representative, swap in the isotope-scaled `K_a`/permeability functions
  already in `transport.py`/`material_library.py`.
- Coolant-side sink is zero-concentration (conservative upper bound). If the
  secondary coolant has appreciable tritium inventory or its own surface
  resistance, replace with a finite `K_eff` BC as in the permeator model.
- Axial `T(z)`: exact counterflow profile from Phase 0; near-linear because the
  capacity rates are balanced (ratio 1.03), so this is no longer a placeholder.
- Tritium depletion of the He stream along `z` is neglected in the first pass —
  check this is a valid approximation (fractional loss << 1) before trusting
  Phase 3 results at face value.
