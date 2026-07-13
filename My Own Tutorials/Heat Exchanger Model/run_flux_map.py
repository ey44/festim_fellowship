"""
run_flux_map.py  —  Phase 2
===========================
Sweep the local permeation flux J_T2(T_wall, p_T2, mdot_He) over the HX
operating envelope and write flux_map.csv for Phase 3.

Grid (edit below):
  T_wall : spans the Phase-0 wall-temperature range (with margin)
  p_T2   : 1 - 100 Pa
  mdot_He: 0.1 - 2 kg/s

backend='analytical' (default) runs anywhere; pass --festim to re-run the
same grid through the FESTIM backend on a machine that has it (slow, but a
direct check that both backends agree over the whole envelope).

Each row also records the regime diagnostics (p_surf/p_bulk, Lambda) so we
know where the operating envelope sits between diffusion- and
surface/film-limited behaviour.
"""

import sys
import warnings
import numpy as np

from hx_1d_model import compute_flux_analytical, compute_flux_festim, DEFAULTS
from material_library import get_material
from transport import Lambda, Lambda_regime

# ── Grid definition ──────────────────────────────────────────────────────────
T_WALL_C = np.linspace(280.0, 600.0, 17)      # C — covers Phase-0 wall range
P_T2     = np.array([1.0, 3.0, 10.0, 30.0, 100.0])   # Pa
MDOT_HE  = np.array([0.1, 0.5, 1.0, 2.0])     # kg/s
F_PHI    = 1.0                                 # bare metal (0.01 = oxidised)


def build_flux_map(backend='analytical', fname='flux_map.csv'):
    mat = get_material(DEFAULTS['material'])
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter('once')          # T-range warning once, not 340x
        for T_c in T_WALL_C:
            T = T_c + 273.15
            for p in P_T2:
                for md in MDOT_HE:
                    r = compute_flux_analytical(T, p, mdot_he=md, f_phi=F_PHI)
                    if backend == 'festim':
                        J = compute_flux_festim(T, p, mdot_he=md, f_phi=F_PHI)
                    else:
                        J = r['J_T2']
                    lam = Lambda(mat, T, p, DEFAULTS['t_wall'],
                                 F_PHI, species='T2')
                    rows.append((T_c, p, md, J, r['J_T2_sieverts'],
                                 r['p_surf_over_p_bulk'], r['K_eff'],
                                 r['Re'], lam))

    header = ('T_wall_C,p_T2_Pa,mdot_he_kg_s,J_T2_mol_m2_s,'
              'J_sieverts_mol_m2_s,p_surf_over_p_bulk,K_eff_mol_m2_s_Pa,'
              'Re_tube,Lambda')
    np.savetxt(fname, np.array(rows), delimiter=',', header=header,
               comments='')
    print(f"{len(rows)} points -> {fname}  (backend: {backend})")
    return np.array(rows)


def summarize(rows):
    """Print scaling checks expected from the physics."""
    mat = get_material(DEFAULTS['material'])
    print("\nSanity checks")
    print("-------------")
    # sqrt(p) scaling at fixed T, mdot
    sel = rows[(rows[:, 0] == 600.0) & (rows[:, 2] == 1.0)]
    J1, J100 = sel[sel[:, 1] == 1.0][0, 3], sel[sel[:, 1] == 100.0][0, 3]
    print(f"  J(100 Pa)/J(1 Pa) at 600 C = {J100/J1:.2f}  (expect ~10 if "
          f"sqrt-law / diffusion-limited)")
    # flow-rate sensitivity
    sel = rows[(rows[:, 0] == 600.0) & (rows[:, 1] == 10.0)]
    Jlo, Jhi = sel[sel[:, 2] == 0.1][0, 3], sel[sel[:, 2] == 2.0][0, 3]
    print(f"  J(2 kg/s)/J(0.1 kg/s) at 600 C, 10 Pa = {Jhi/Jlo:.4f}  "
          f"(~1 -> gas film negligible, flow rate barely matters)")
    # temperature leverage
    sel = rows[(rows[:, 1] == 10.0) & (rows[:, 2] == 1.0)]
    Jc = sel[sel[:, 0] == 280.0][0, 3]
    Jh = sel[sel[:, 0] == 600.0][0, 3]
    print(f"  J(600 C)/J(280 C) = {Jh/Jc:.0f}  -> permeation lives at the "
          f"hot end")
    # regime
    lam_min, lam_max = rows[:, 8].min(), rows[:, 8].max()
    print(f"  Lambda range = {lam_min:.1e} - {lam_max:.1e}  "
          f"({Lambda_regime(lam_min)} to {Lambda_regime(lam_max)}; "
          f"NB Pd-proxy K_a — indicative only for IN617)")


if __name__ == '__main__':
    backend = 'festim' if '--festim' in sys.argv else 'analytical'
    rows = build_flux_map(backend=backend)
    summarize(rows)
