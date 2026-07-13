"""
hx_2d_model.py  —  Phase 4
==========================
2D FESTIM model (axial z x through-wall y) of the whole heat exchanger
wall, to check the "stitched 1D" approximation of Phase 3.

RUN THIS ON YOUR FESTIM MACHINE (needs festim 2.x + dolfinx).

Design choice vs the original plan: instead of solving a coupled
F.HeatTransferProblem, the wall temperature field T(z, y) is PRESCRIBED
from the Phase-0 thermal solution (He-side and steam-side wall surface
temperatures, linear through the thin wall). For a 2.5 mm wall with known
film coefficients this is essentially exact, removes the most fragile part
of the coupling, and still tests everything Phase 3 approximates:
  - axial conduction/diffusion in the wall (absent from stitched-1D),
  - the nonlinearity of J(T) sampled continuously instead of station-wise,
  - the spatially varying Sieverts BC.
A fully coupled heat solve can be added later by swapping the `temperature`
argument for a HeatTransferProblem solution.

Output: total Q_T [mol T2/s] from SurfaceFlux on the steam-side boundary
(x perimeter), compared against the Phase-3 integrated value.
"""

import numpy as np

from material_library import get_material, M_T, M_H
from hx_thermal_sizing import size_hx
from axial_integration import integrate_hx

KJMOL_TO_EV = 1.0 / 96.485

# ── Operating point (match Phase 3 baseline) ─────────────────────────────────
P_T2    = 10.0      # Pa
MDOT_HE = 1.0       # kg/s
F_PHI   = 1.0
NZ, NY  = 400, 16   # mesh divisions (axial, through-wall)


def run_2d():
    import festim as F
    import dolfinx
    from mpi4py import MPI

    # ── Phase-0 geometry and wall-temperature profiles ──
    design, prof = size_hx(mdot_he=MDOT_HE, n_z=201, verbose=False)
    L      = design['L_m']
    t_w    = design['inputs']['t_wall']
    perim  = design['n_tubes'] * np.pi * design['inputs']['d_i']

    # ── Closed-form T(z, y) in UFL-safe operations ──
    # FESTIM passes a SYMBOLIC (UFL) coordinate to the temperature callable,
    # so table interpolation (np.interp) is not possible. The Phase-0
    # counterflow profile is analytic, so we rebuild it symbolically:
    #   a(z)        = (A/L) z                      cumulative area
    #   dT(z)       = dT0 exp(-r a)                local He-steam dT
    #   T_he(z)     = T_h_in - (U/C_h) dT0 (1-exp(-r a))/r
    #   T_wall(z,y) = T_he - (U dT/h_he) - (U dT/k_wall) y
    import ufl

    inp   = design['inputs']
    U     = design['U_W_m2K']
    A     = design['A_m2']
    C_h   = design['C_hot_W_K']
    C_c   = design['C_cold_W_K']
    h_he  = design['h_he_W_m2K']
    k_w   = inp['k_wall']
    dT0   = inp['T_h_in'] - inp['T_c_out']
    r_cf  = U * (1.0 / C_h - 1.0 / C_c)            # 1/m2 (negative here)

    def temperature(x):
        a   = (A / L) * x[0]
        dT  = dT0 * ufl.exp(-r_cf * a)
        T_he = inp['T_h_in'] - (U / C_h) * dT0 * (1 - ufl.exp(-r_cf * a)) / r_cf
        return T_he - (U * dT / h_he) - (U * dT / k_w) * x[1]

    # numerical cross-check of the symbolic formula vs the Phase-0 arrays
    z_p = prof['z_m']
    a_p = (A / L) * z_p
    dT_p = dT0 * np.exp(-r_cf * a_p)
    T_he_p = inp['T_h_in'] - (U / C_h) * dT0 * (1 - np.exp(-r_cf * a_p)) / r_cf
    T_w_p = T_he_p - (U * dT_p / h_he)
    err = np.max(np.abs(T_w_p - prof['T_wall_he_K']))
    assert err < 1e-6, f"T(z) closed form deviates from Phase 0 ({err=} K)"

    # ── Material (tritium-scaled Inconel 617) ──
    from hx_1d_model import make_festim_material
    mat = get_material('Inconel617')
    inconel = make_festim_material(
        D_0=mat['D0'] / np.sqrt(M_T / M_H),
        E_D=mat['ED'] * KJMOL_TO_EV,
    )
    S_0 = 2.0 * mat['KS0'] * F_PHI            # mol T / m3 / Pa^0.5 (atomic)
    E_S = mat['ES'] * KJMOL_TO_EV

    # ── Mesh: rectangle [0,L] x [0,t_w] ──
    fenics_mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [[0.0, 0.0], [L, t_w]], [NZ, NY],
        cell_type=dolfinx.mesh.CellType.quadrilateral,
    )
    mesh = F.Mesh(mesh=fenics_mesh)

    vol   = F.VolumeSubdomain(id=1, material=inconel,
                              locator=lambda x: np.full_like(x[0], True, dtype=bool))
    s_he  = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[1], 0.0))
    s_sec = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[1], t_w))

    tritium = F.Species('T')

    bcs = [
        F.SievertsBC(subdomain=s_he, S_0=S_0, E_S=E_S,
                     pressure=P_T2, species=tritium),
        F.FixedConcentrationBC(subdomain=s_sec, value=0.0, species=tritium),
    ]

    flux_out = F.SurfaceFlux(field=tritium, surface=s_sec)

    model = F.HydrogenTransportProblem(
        mesh=mesh,
        subdomains=[vol, s_he, s_sec],
        species=[tritium],
        boundary_conditions=bcs,
        exports=[flux_out],
        temperature=temperature,
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )
    model.initialise()
    model.run()

    # SurfaceFlux in 2D = line integral of the normal flux along the
    # steam-side edge (length L), per unit depth: [mol T / (m depth) / s].
    # The slab's depth direction is the unrolled tube circumference, so the
    # total depth is the wetted perimeter N*pi*d_i:
    J_line_atomic = abs(flux_out.data[-1])
    Q_T2 = 0.5 * J_line_atomic * perim          # mol T2 / s

    return Q_T2, design


if __name__ == '__main__':
    try:
        import festim  # noqa: F401
    except ImportError:
        raise SystemExit("FESTIM/dolfinx not available in this environment — "
                         "run on your FESTIM machine.")

    Q_2d, design = run_2d()
    print(f"Phase 4 (2D FESTIM, slab):    Q_T = {Q_2d:.4e} mol T2/s")

    # like-for-like check: the 2D model is a Cartesian slab, so compare it
    # against Phase 3 run in slab mode — this isolates the stitched-1D
    # approximation from the (separately exact) curvature correction.
    r3s = integrate_hx(p_T2_in=P_T2, mdot_he=MDOT_HE, f_phi=F_PHI,
                       geometry='slab', quiet=True)
    print(f"Phase 3 (stitched 1D, slab):  Q_T = {r3s['Q_T2_mol_s']:.4e} mol T2/s")
    print(f"Relative difference (tests the stitching only): "
          f"{abs(Q_2d - r3s['Q_T2_mol_s'])/r3s['Q_T2_mol_s']:.2%}")

    r3c = integrate_hx(p_T2_in=P_T2, mdot_he=MDOT_HE, f_phi=F_PHI, quiet=True)
    print(f"\nProduction number (Phase 3, cylindrical wall): "
          f"Q_T = {r3c['Q_T2_mol_s']:.4e} mol T2/s "
          f"= {r3c['Ci_per_day']:,.0f} Ci/day")
    print("\nIf the slab-vs-slab discrepancy is more than a few %, suspects: "
          "axial diffusion at the hot end (steep J(z)), mesh resolution near "
          "z=0, or the station count in Phase 3.")
