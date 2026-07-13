"""
hx_1d_model.py  —  Phase 1
==========================
1D through-wall steady-state tritium permeation at a single operating point
(T, p_T2, Q_He), Inconel 617 wall, with two interchangeable backends:

  backend='analytical' : closed-form series-resistance solution.
                         Runs anywhere (no FESTIM needed). Used by the
                         Phase-2 sweep.
  backend='festim'     : FESTIM 2.x 1D model (F.HydrogenTransportProblem),
                         SievertsBC on the He side, zero-concentration sink
                         on the steam side. Run this on a machine with
                         FESTIM installed — it is the verification of the
                         analytical backend (see validate_festim()).

Physics
-------
He side:    series resistance  1/K_eff = RT/k_m  (+ 1/K_a if enabled)
            J_T2 = K_eff * (p_bulk - p_surf)          [linear in p]
Wall:       Sieverts + Fickian diffusion, zero sink:
            J_T2 = Phi_T2(T)/t_wall * sqrt(p_surf)    [Richardson]
Solving the balance for s = sqrt(p_surf):
            K_eff s^2 + (Phi/t) s - K_eff p_bulk = 0  ->  closed form.
With K_eff -> inf this reduces to the pure-Sieverts result
            J = Phi_T2 * sqrt(p_bulk) / t_wall.

Units: molecular (T2) fluxes in mol/m2/s everywhere in this module.
FESTIM internally tracks ATOMIC concentration (mol T / m3): S_0 = 2*K_S,
and its SurfaceFlux is divided by 2 on return.

Surface kinetics caveat: K_a data for Inconel 617 do not exist; the
Vadrucci Pd/Ag value in transport.py is NOT transferable. By default only
the gas-film resistance is included (include_surface_kinetics=False); the
Pd-proxy option exists purely as a sensitivity lever.
"""

import numpy as np

from material_library import (
    get_material, sieverts_constant, permeability_species,
    diffusivity_isotope, M_T, M_H,
)
from transport import mass_transfer_coeff, surface_resistance_H, R_GAS

KJMOL_TO_EV = 1.0 / 96.485  # 1 kJ/mol = 0.010364 eV

# ── Defaults (overridden by callers) ─────────────────────────────────────────
DEFAULTS = dict(
    material   = 'Inconel617',
    t_wall     = 2.5e-3,    # m
    P_he       = 2e5,       # Pa, total He pressure
    d_i        = 0.010,     # m, tube ID (gas-film correlation)
    n_tubes    = 1183,      # from hx_thermal_sizing baseline
    f_phi      = 1.0,       # permeability factor (0.01 ~ oxidised IN617)
    include_surface_kinetics = False,
    geometry   = 'cylindrical',   # 'cylindrical' (exact) or 'slab'
)


def effective_thickness(cfg):
    """
    Through-wall diffusion length referenced to the INNER surface.

    slab        : t_wall
    cylindrical : r_i * ln(r_o/r_i)  — exact for steady diffusion through an
                  annulus. With it, J(inner surface) * inner perimeter equals
                  the exact cylindrical permeation rate, so downstream code
                  can keep using the inner perimeter unchanged.
                  For r_i=5 mm, t=2.5 mm: t_eff = 2.03 mm (J up ~23% vs slab).
    """
    if cfg['geometry'] == 'slab':
        return cfg['t_wall']
    r_i = cfg['d_i'] / 2.0
    r_o = r_i + cfg['t_wall']
    return r_i * np.log(r_o / r_i)


def make_festim_material(**kw):
    """F.Material across FESTIM 2.x variants (some require an `id` arg)."""
    import festim as F
    try:
        return F.Material(**kw)
    except TypeError:
        return F.Material(id=1, **kw)


def kg_s_to_nlpm(mdot_he):
    """He mass flow kg/s -> normal litres/min (transport.py convention)."""
    mol_s = mdot_he / 4.003e-3
    return mol_s * 22.414e-3 * 1000.0 * 60.0


def K_eff_he_side(T, p=None, mdot_he=1.0, **kw):
    """
    Combined He-side resistance coefficient K_eff [mol(T2)/m2/s/Pa].
    Gas film always; surface kinetics optional (Pd proxy — see module note).
    """
    cfg = {**DEFAULTS, **kw}
    mt = mass_transfer_coeff(T, cfg['P_he'], kg_s_to_nlpm(mdot_he),
                             cfg['d_i'], cfg['n_tubes'], species='T2')
    R_total = mt['R_gas_film']                       # RT/k_m  [m2.s.Pa/mol]
    if cfg['include_surface_kinetics']:
        R_total = R_total + surface_resistance_H(T)  # Pd proxy!
    return {'K_eff': 1.0 / R_total, 'k_m': mt['k_m'], 'Re': mt['Re'],
            'Sh': mt['Sh'], 'regime': mt['regime'],
            'R_gas_film': mt['R_gas_film']}


# ── Analytical backend ───────────────────────────────────────────────────────

def compute_flux_analytical(T, p_T2, mdot_he=1.0, **kw):
    """
    Steady tritium permeation flux [mol(T2)/m2/s] through the wall at local
    wall temperature T [K] and He-side bulk partial pressure p_T2 [Pa].

    The flux is referenced to the INNER (He-side) surface; multiply by the
    inner perimeter for totals. With geometry='cylindrical' (default) the
    annular geometry is exact via effective_thickness().

    Returns dict with J_T2, p_surf, and diagnostics.
    """
    cfg = {**DEFAULTS, **kw}
    mat = get_material(cfg['material'])

    Phi_T2 = permeability_species(mat, T, 'T2', cfg['f_phi'])  # mol/m/s/Pa^0.5
    k_diff = Phi_T2 / effective_thickness(cfg)                 # mol/m2/s/Pa^0.5

    he = K_eff_he_side(T, mdot_he=mdot_he, **kw)
    K_eff = he['K_eff']

    # K_eff s^2 + k_diff s - K_eff p_T2 = 0,  s = sqrt(p_surf) >= 0
    s = (-k_diff + np.sqrt(k_diff**2 + 4 * K_eff**2 * p_T2)) / (2 * K_eff)
    p_surf = s ** 2
    J_T2 = k_diff * s

    J_sieverts = k_diff * np.sqrt(p_T2)   # K_eff -> inf limit
    return {
        'J_T2': J_T2,                     # mol T2 / m2 / s
        'J_T2_sieverts': J_sieverts,      # pure-Sieverts (upper) limit
        'p_surf': p_surf,
        'p_surf_over_p_bulk': p_surf / p_T2 if p_T2 > 0 else 1.0,
        'Phi_T2': Phi_T2,
        'K_eff': K_eff,
        'Re': he['Re'], 'gas_regime': he['regime'],
    }


# ── FESTIM backend ───────────────────────────────────────────────────────────

def compute_flux_festim(T, p_T2, n_mesh=201, **kw):
    """
    Same operating point solved with FESTIM 2.x. Requires festim+dolfinx.
    Pure-Sieverts He-side BC at effective surface pressure p_surf taken from
    the analytical He-side balance (exact at steady state), so the FESTIM
    run verifies the WALL part (diffusion + Sieverts + sink) of the model.
    Set use_bulk_pressure=True to apply p_T2 directly (pure-Sieverts case).

    Returns J_T2 [mol(T2)/m2/s] from F.SurfaceFlux on the steam-side face.
    """
    import festim as F

    cfg = {**DEFAULTS, **kw}
    use_bulk = kw.get('use_bulk_pressure', False)
    mat_dict = get_material(cfg['material'])

    # pressure applied at the metal surface
    p_apply = p_T2 if use_bulk else compute_flux_analytical(T, p_T2, **kw)['p_surf']

    # FESTIM material: tritium diffusivity (isotope-scaled), eV units
    D_0_T = mat_dict['D0'] / np.sqrt(M_T / M_H)
    E_D   = mat_dict['ED'] * KJMOL_TO_EV
    inconel = make_festim_material(D_0=D_0_T, E_D=E_D)

    # Sieverts: atomic concentration  c = S_0 sqrt(p),  S_0 = 2*K_S  [mol T/m3/Pa^0.5]
    S_0 = 2.0 * mat_dict['KS0'] * cfg['f_phi']   # f_phi applied to solubility
    E_S = mat_dict['ES'] * KJMOL_TO_EV

    t_w  = cfg['t_wall']
    mesh = F.Mesh1D(vertices=np.linspace(0.0, t_w, n_mesh))
    vol  = F.VolumeSubdomain1D(id=1, borders=[0.0, t_w], material=inconel)
    s_he  = F.SurfaceSubdomain1D(id=2, x=0.0)     # He side
    s_sec = F.SurfaceSubdomain1D(id=3, x=t_w)     # steam side

    tritium = F.Species('T')

    bcs = [
        F.SievertsBC(subdomain=s_he, S_0=S_0, E_S=E_S,
                     pressure=p_apply, species=tritium),
        F.FixedConcentrationBC(subdomain=s_sec, value=0.0, species=tritium),
    ]

    flux_out = F.SurfaceFlux(field=tritium, surface=s_sec)

    model = F.HydrogenTransportProblem(
        mesh=mesh,
        subdomains=[vol, s_he, s_sec],
        species=[tritium],
        boundary_conditions=bcs,
        exports=[flux_out],
        temperature=float(T),
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )
    model.initialise()
    model.run()

    J_atomic = abs(flux_out.data[-1])     # mol T / m2 / s
    return J_atomic / 2.0                 # mol T2 / m2 / s


def compute_flux_festim_cylindrical(T, p_T2, n_mesh=201, **kw):
    """
    Cylindrical 1D FESTIM solve across the annular wall (r_i -> r_o), pure
    Sieverts at r_i, zero sink at r_o. Verifies the exact annulus result
        Q/len = 2*pi*Phi_T2*sqrt(p) / ln(r_o/r_i)
    i.e. the effective_thickness() correction.

    FESTIM's SurfaceFlux export is Cartesian-only, so the flux is extracted
    manually from the solved concentration profile: at steady state
    Q/len = -2*pi*r*D*dc/dr is constant in r; we evaluate it across the wall
    and check flatness.

    Returns J_T2 referenced to the INNER surface [mol(T2)/m2/s].
    """
    import festim as F

    cfg = {**DEFAULTS, **kw}
    mat_dict = get_material(cfg['material'])
    r_i = cfg['d_i'] / 2.0
    r_o = r_i + cfg['t_wall']

    D_0_T = mat_dict['D0'] / np.sqrt(M_T / M_H)
    E_D   = mat_dict['ED'] * KJMOL_TO_EV
    S_0   = 2.0 * mat_dict['KS0'] * cfg['f_phi']
    E_S   = mat_dict['ES'] * KJMOL_TO_EV

    mesh = F.Mesh1D(vertices=np.linspace(r_i, r_o, n_mesh),
                    coordinate_system='cylindrical')
    vol  = F.VolumeSubdomain1D(id=1, borders=[r_i, r_o],
                               material=make_festim_material(D_0=D_0_T, E_D=E_D))
    s_in  = F.SurfaceSubdomain1D(id=2, x=r_i)
    s_out = F.SurfaceSubdomain1D(id=3, x=r_o)

    tritium = F.Species('T')
    model = F.HydrogenTransportProblem(
        mesh=mesh,
        subdomains=[vol, s_in, s_out],
        species=[tritium],
        boundary_conditions=[
            F.SievertsBC(subdomain=s_in, S_0=S_0, E_S=E_S,
                         pressure=p_T2, species=tritium),
            F.FixedConcentrationBC(subdomain=s_out, value=0.0, species=tritium),
        ],
        temperature=float(T),
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )
    model.initialise()
    model.run()

    # manual flux extraction: Q/len = -2*pi*r*D*dc/dr (constant in r)
    c_sol = tritium.post_processing_solution
    r = c_sol.function_space.mesh.geometry.x[:, 0]
    c = c_sol.x.array[:]
    idx = np.argsort(r); r, c = r[idx], c[idx]

    kB_eV = 8.617e-5
    D = D_0_T * np.exp(-E_D / (kB_eV * T))
    Q_len = -2.0 * np.pi * r * D * np.gradient(c, r)   # mol T / m / s

    interior = Q_len[2:-2]                              # avoid edge stencils
    spread = (interior.max() - interior.min()) / interior.mean()
    if abs(spread) > 1e-2:
        print(f"  WARNING: 2*pi*r*D*dc/dr not constant (spread {spread:.1e}) "
              f"— refine mesh?")

    J_atomic_inner = interior.mean() / (2.0 * np.pi * r_i)
    return J_atomic_inner / 2.0                         # mol T2 / m2 / s


# ── Validation (run on a machine with FESTIM) ────────────────────────────────

def validate_festim():
    """
    Phase-1 V&V:
      1. Slab: FESTIM (Cartesian) vs analytical J = Phi*sqrt(p)/t, three T's.
      2. Mesh convergence at the hottest point (slab).
      3. Cylindrical: FESTIM (cylindrical 1D, manual flux extraction) vs the
         exact annulus solution with t_eff = r_i*ln(r_o/r_i).
    """
    p_T2 = 10.0
    print("Slab geometry:")
    print(f"{'T [C]':>7} {'J analytical':>14} {'J FESTIM':>14} {'rel.err':>9}")
    for T_c in (580.0, 450.0, 300.0):
        T = T_c + 273.15
        Ja = compute_flux_analytical(T, p_T2, geometry='slab')['J_T2_sieverts']
        Jf = compute_flux_festim(T, p_T2, use_bulk_pressure=True)
        print(f"{T_c:7.0f} {Ja:14.4e} {Jf:14.4e} {abs(Jf-Ja)/Ja:9.2e}")

    print("\nMesh convergence (slab, T=580 C):")
    T = 580 + 273.15
    Ja = compute_flux_analytical(T, p_T2, geometry='slab')['J_T2_sieverts']
    for n in (11, 51, 201, 801):
        Jf = compute_flux_festim(T, p_T2, n_mesh=n, use_bulk_pressure=True)
        print(f"  n={n:4d}  J={Jf:.6e}  rel.err={abs(Jf-Ja)/Ja:.2e}")

    print("\nCylindrical geometry (exact annulus, J at inner surface):")
    print(f"{'T [C]':>7} {'J analytical':>14} {'J FESTIM':>14} {'rel.err':>9}")
    for T_c in (580.0, 450.0, 300.0):
        T = T_c + 273.15
        Ja = compute_flux_analytical(T, p_T2)['J_T2_sieverts']  # cylindrical default
        Jf = compute_flux_festim_cylindrical(T, p_T2)
        print(f"{T_c:7.0f} {Ja:14.4e} {Jf:14.4e} {abs(Jf-Ja)/Ja:9.2e}")


if __name__ == '__main__':
    # Analytical spot checks (always runnable)
    print("Analytical backend — baseline p_T2=10 Pa, mdot_He=1 kg/s, bare metal:")
    print(f"{'T_wall [C]':>11} {'J_T2 [mol/m2/s]':>17} {'p_s/p_b':>9} "
          f"{'limit J(Sieverts)':>18}")
    for T_c in (580, 500, 400, 300):
        r = compute_flux_analytical(T_c + 273.15, 10.0)
        print(f"{T_c:11.0f} {r['J_T2']:17.4e} {r['p_surf_over_p_bulk']:9.4f} "
              f"{r['J_T2_sieverts']:18.4e}")

    print("\nWith oxide suppression f_phi=0.01:")
    r = compute_flux_analytical(580 + 273.15, 10.0, f_phi=0.01)
    print(f"  J_T2(580 C) = {r['J_T2']:.4e} mol/m2/s")

    # FESTIM validation if available
    try:
        import festim  # noqa: F401
        print("\nFESTIM found — running Phase-1 validation:")
        validate_festim()
    except ImportError:
        print("\nFESTIM not installed here — run validate_festim() on your "
              "FESTIM machine to complete Phase-1 V&V.")
