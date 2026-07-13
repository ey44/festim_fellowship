"""
axial_integration.py  —  Phase 3
================================
Combine the Phase-0 geometry/temperature profiles with the Phase-1 local
flux model to get the total tritium permeation rate Q_T into the secondary
coolant:

    Q_T = integral over z of  J_T2(T_wall(z), p_T2(z)) * N * pi * d_i  dz

- T_wall(z): He-side wall surface temperature from hx_thermal_sizing
  (NOT the gas temperature — with the He film dominating U, the wall sits
  much closer to the secondary temperature).
- p_T2(z): either constant (default) or depleted self-consistently along z
  (dF_T2/dz = -J * N * pi * d_i); both are computed and compared so the
  "depletion negligible" assumption is checked, not assumed.
- J is evaluated directly from the analytical model (cheap), so no
  interpolation error; flux_map.csv remains the FESTIM cross-check artifact.

Outputs: baseline Q_T in mol/s, g/day, Ci/day; convergence in n_z;
sensitivity tables over p_T2, mdot_He, f_phi, and the U-driven area band.

Run:  python3 axial_integration.py   (requires hx_design.json/hx_profiles.csv
      from hx_thermal_sizing.py; regenerates them if missing)
"""

import warnings
import numpy as np

from hx_1d_model import compute_flux_analytical
from hx_thermal_sizing import size_hx

trapz = getattr(np, 'trapezoid', None) or np.trapz   # numpy 2.x / 1.x

# tritium activity constants
T_HALF_S   = 12.32 * 3.1557e7          # s
LAMBDA_T   = np.log(2.0) / T_HALF_S    # 1/s
N_AVOGADRO = 6.02214e23
BQ_PER_MOL_T = LAMBDA_T * N_AVOGADRO   # ~1.074e15 Bq per mol T atoms
CI         = 3.7e10                    # Bq
M_T_G      = 3.016                     # g/mol T atoms
M_HE       = 4.003e-3                  # kg/mol


def integrate_hx(p_T2_in=10.0, mdot_he=1.0, f_phi=1.0, n_z=101,
                 deplete=True, design_kw=None, quiet=False,
                 geometry='cylindrical'):
    """
    Returns dict with Q_T (mol T2 /s), activity rates, depletion fraction.

    geometry: 'cylindrical' (exact annular wall, default) or 'slab'
    (flat-wall approximation, ~19% lower — kept for the 2D FESTIM check,
    which is a Cartesian slab).
    """
    design_kw = design_kw or {}
    design, prof = size_hx(mdot_he=mdot_he, n_z=n_z, verbose=False,
                           **design_kw)
    z      = prof['z_m']
    T_wall = prof['T_wall_he_K']
    N      = design['n_tubes']
    d_i    = design['inputs']['d_i']
    perim  = N * np.pi * d_i                       # total wetted perimeter [m]

    F_T2_in = (p_T2_in / design['inputs']['P_he']) * mdot_he / M_HE  # mol T2/s

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')            # T-range warned elsewhere

        if deplete:
            # march downstream, updating p_T2 from the molar balance
            p = np.empty_like(z); J = np.empty_like(z)
            p[0] = p_T2_in
            F = F_T2_in
            for i in range(len(z)):
                J[i] = compute_flux_analytical(
                    T_wall[i], max(p[i if i < len(z) else -1], 0.0),
                    mdot_he=mdot_he, f_phi=f_phi, n_tubes=N,
                    geometry=geometry)['J_T2']
                if i < len(z) - 1:
                    dz = z[i + 1] - z[i]
                    F = max(F - J[i] * perim * dz, 0.0)
                    p[i + 1] = p_T2_in * F / F_T2_in
        else:
            p = np.full_like(z, p_T2_in)
            J = np.array([compute_flux_analytical(
                T, p_T2_in, mdot_he=mdot_he, f_phi=f_phi,
                n_tubes=N, geometry=geometry)['J_T2'] for T in T_wall])

    Q_T2 = trapz(J * perim, z)                  # mol T2 / s
    frac_depleted = Q_T2 / F_T2_in

    mol_T_s  = 2.0 * Q_T2
    g_day    = mol_T_s * M_T_G * 86400.0
    ci_day   = mol_T_s * BQ_PER_MOL_T / CI * 86400.0

    out = {
        'Q_T2_mol_s': Q_T2, 'g_T_per_day': g_day, 'Ci_per_day': ci_day,
        'frac_of_inlet_T_permeated': frac_depleted,
        'A_m2': design['A_m2'], 'U_W_m2K': design['U_W_m2K'],
        'J_profile': J, 'p_profile': p, 'z': z, 'T_wall': T_wall,
    }
    if not quiet:
        print(f"  Q_T = {Q_T2:.3e} mol T2/s  =  {g_day:.3f} g T/day  "
              f"=  {ci_day:,.0f} Ci/day   "
              f"(depleted fraction of inlet T: {frac_depleted:.2%})")
    return out


if __name__ == '__main__':
    print("=" * 64)
    print("Phase 3 — total tritium permeation (baseline)")
    print("=" * 64)
    print("Baseline: p_T2=10 Pa, mdot_He=1 kg/s, bare Inconel 617 "
          "(cylindrical wall):")
    base = integrate_hx()

    slab = integrate_hx(geometry='slab', quiet=True)
    print(f"  (slab approximation would give {slab['Q_T2_mol_s']:.3e} mol/s "
          f"— {base['Q_T2_mol_s']/slab['Q_T2_mol_s']:.1%} of cylindrical; "
          f"curvature adds ~{base['Q_T2_mol_s']/slab['Q_T2_mol_s']-1:.0%})")

    print("\nDepletion check (constant-p vs depleted):")
    nodep = integrate_hx(deplete=False, quiet=True)
    print(f"  constant-p Q_T = {nodep['Q_T2_mol_s']:.3e}, "
          f"depleted Q_T = {base['Q_T2_mol_s']:.3e}  "
          f"(diff {abs(1 - base['Q_T2_mol_s']/nodep['Q_T2_mol_s']):.2%})")

    print("\nAxial convergence:")
    for n in (11, 26, 51, 101, 201):
        r = integrate_hx(n_z=n, quiet=True)
        print(f"  n_z={n:4d}  Q_T={r['Q_T2_mol_s']:.5e} mol/s")

    print("\nWhere does it permeate? (cumulative Q_T along z, baseline)")
    J, z = base['J_profile'], base['z']
    perim_int = np.array([trapz(J[:i+1], z[:i+1]) for i in range(len(z))])
    for fr in (0.5, 0.8, 0.9):
        i = np.searchsorted(perim_int, fr * perim_int[-1])
        print(f"  {fr:.0%} of permeation occurs in the first "
              f"{z[i]/z[-1]:.0%} of the tube (hot end)")

    print("\nSensitivity — p_T2 [Pa] (mdot=1, bare):")
    for p in (1.0, 3.0, 10.0, 30.0, 100.0):
        r = integrate_hx(p_T2_in=p, quiet=True)
        print(f"  p_T2={p:6.1f}  Q_T={r['Q_T2_mol_s']:.3e} mol/s  "
              f"{r['Ci_per_day']:>12,.0f} Ci/day")

    print("\nSensitivity — mdot_He [kg/s] (p=10 Pa; geometry RE-SIZED for "
          "same duty proportions):")
    for md in (0.1, 0.5, 1.0, 2.0):
        r = integrate_hx(mdot_he=md, quiet=True)
        print(f"  mdot={md:4.1f}  A={r['A_m2']:7.1f} m2  "
              f"Q_T={r['Q_T2_mol_s']:.3e} mol/s  "
              f"{r['Ci_per_day']:>12,.0f} Ci/day")

    print("\nSensitivity — oxide factor f_phi:")
    for f in (1.0, 0.1, 0.01):
        r = integrate_hx(f_phi=f, quiet=True)
        print(f"  f_phi={f:5.2f}  Q_T={r['Q_T2_mol_s']:.3e} mol/s  "
              f"{r['Ci_per_day']:>12,.0f} Ci/day")

    print("\nSensitivity — U/area band (via Re_target & h_sec):")
    for re_t, hs, tag in ((2000.0, 2000.0, 'low-U / big-A'),
                          (3000.0, 4000.0, 'baseline'),
                          (8000.0, 8000.0, 'high-U / small-A')):
        r = integrate_hx(design_kw=dict(Re_target=re_t, h_sec=hs), quiet=True)
        print(f"  {tag:18s} U={r['U_W_m2K']:6.1f}  A={r['A_m2']:7.1f} m2  "
              f"Q_T={r['Q_T2_mol_s']:.3e} mol/s  "
              f"{r['Ci_per_day']:>12,.0f} Ci/day")
