"""
permeator_v2.py
===============
1D steady-state He purge loop Pd/Ag permeator model.
H2/T2 binary feed (no HT).

Unit system: MOLECULAR throughout
----------------------------------
All fluxes J [mol molecules/m²/s].
All molar flows F [mol molecules/s].
K_a, K_eff_f in mol molecules/m²/s/Pa  (as given by Vadrucci/Shimada literature).
Phi in mol(H2)/m/s/Pa^0.5              (as given by Serra/Shimada).

Governing equation (implicit, solved at each z):

    J_mol = (Phi_eff / delta) * ( sqrt(P_H_total - J_mol / K_eff_f)
                                  - sqrt(J_mol / K_a_s + p_perm)  )

where:
    Phi_eff(z) = x_H * Phi_H + x_T * Phi_T     [mol/m/s/Pa^0.5]
    1/K_eff_f  = RT/k_m + 1/K_a_f              [m²·s·Pa/mol]
    J_mol      = total molecular flux (H2 + T2) [mol molecules/m²/s]

Individual species molecular fluxes:
    J_H2 = J_mol * x_H * Phi_H / Phi_eff
    J_T2 = J_mol * x_T * Phi_T / Phi_eff

ODE state variables: F_H2(z), F_T2(z)  [mol molecules/s, total bundle]
    dF_H2/dz = -J_H2 * pi * d_i * n_tubes
    dF_T2/dz = -J_T2 * pi * d_i * n_tubes

No factor of 2 anywhere: the factor 2 that converts between mol(H2) and
mol(H atoms) cancels exactly when going from atomic Fick's law back to
molecular flux (J_H2_mol = J_H_atoms/2 = Phi/delta * sqrt(p)).

Regime diagnostic:
    Lambda(z) = (Phi_eff/delta) / (K_eff_f * sqrt(P_H_total))
              = J_diff_lim / J_surf_lim   [dimensionless, molecular units]
    Lambda > 1 -> surface-limited
    Lambda < 1 -> diffusion-limited

References
----------
Glugla et al.   J. Nucl. Mater. 355 (2006) 47-53
Shimada         Comprehensive Nuclear Materials 2nd ed. (2020) ch. 6.08
Antunes et al.  Fusion Sci. Technol. (2020)
Vadrucci et al. Int. J. Hydrogen Energy 38 (2013) 4144
Chung & Dalgarno Phys. Rev. A 66 (2002) 012712
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import warnings

from material_library import (
    get_material, permeability_species, sieverts_constant,
    M_H, M_T, M_He,
)
from transport import (
    mass_transfer_coeff, surface_resistance_H, surface_resistance_T,
    R_GAS,
)

# ── Default parameters ────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    # Geometry
    'd_i':       0.010,     # m   tube inner diameter
    'delta':     100e-6,    # m   membrane thickness
    'L':         0.500,     # m   active tube length
    'n_tubes':   100,        # -   parallel tubes

    # Operating point
    'T':         623.0,     # K
    'P_He':      2e5,       # Pa  total pressure (He dominant)
    'p_perm':    0.01,       # Pa  permeate back-pressure

    # Feed: specify EITHER F_H_total OR Q_He_nlpm — whichever is more natural.
    # If F_H_total is given, F_He is derived from p_H_total and F_H_total.
    # If Q_He_nlpm is given and F_H_total is None, F_H_total is derived
    # from Q_He_nlpm and p_H_total (legacy behaviour).
    'F_H_total': 7e-4,      # mol/s  total hydrogen (H2+T2) molar flow  [PRIMARY]
    'Q_He_nlpm': None,      # Nl/min total He flow  [set if F_H_total not given]
    'p_H_total': 100.0,     # Pa  total hydrogen partial pressure at inlet
    'x_H_in':    0.5,       # -   H2/(H2+T2) molecular mole fraction at inlet

    # Material
    'material':   'Pd25Ag',
    'f_phi':      1.0,       # fouling factor [0,1]
    'Ka_scaling': 'equal',   # 'equal' or 'scaled' for T vs H surface kinetics

    # Numerical
    'solver':         'RK45',
    'rtol':           1e-8,
    'atol':           1e-14,   # mol molecules/s
    'newton_maxiter': 50,
    'newton_tol':     1e-14,   # mol molecules/m2/s
    'n_out':          500,
}

SPECIES = ['H2', 'T2']


# ── Parameter validation and preprocessing ───────────────────────────────────

def build_params(user_params: dict) -> dict:
    """
    Merge user overrides with defaults, validate, and precompute all
    spatially-uniform quantities.
    """
    p = {**DEFAULT_PARAMS, **user_params}

    if not (0.0 <= p['x_H_in'] <= 1.0):
        raise ValueError(f"x_H_in must be in [0,1], got {p['x_H_in']}")
    p['x_T_in'] = 1.0 - p['x_H_in']

    # Inlet partial pressures
    p['p_H2_in'] = p['x_H_in'] * p['p_H_total']
    p['p_T2_in'] = p['x_T_in'] * p['p_H_total']

    # Material
    mat = get_material(p['material'])
    p['mat'] = mat

    # Permeabilities [mol(H2)/m/s/Pa^0.5]
    # Phi_H2: permeability for H2 molecules (atomic H diffusion, molecular units)
    # Phi_T2: permeability for T2 molecules (atomic T diffusion, molecular units)
    p['Phi_H2'] = permeability_species(mat, p['T'], 'H2', p['f_phi'])
    p['Phi_T2'] = permeability_species(mat, p['T'], 'T2', p['f_phi'])

    # Sieverts constant
    p['KS'] = sieverts_constant(mat, p['T'])

    # ── Flow rates ────────────────────────────────────────────────────────────
    # Mode A (primary): F_H_total given → derive F_He from p_H_total
    # Mode B (legacy):  Q_He_nlpm given → derive F_H_total from p_H_total
    T_n, P_n = 273.15, 101325.0

    if p.get('F_H_total') is not None:
        p['F_H_total_in'] = p['F_H_total']
        p['F_He_total']   = p['F_H_total'] * (p['P_He'] - p['p_H_total']) / p['p_H_total']
        p['Q_He_nlpm']    = p['F_He_total'] * R_GAS * T_n / P_n * 1e3 * 60
    else:
        Q_act = (p['Q_He_nlpm'] / 60.0) * 1e-3 * (p['T'] / T_n) * (P_n / p['P_He'])
        rho_He = p['P_He'] * (M_He * 1e-3) / (R_GAS * p['T'])
        p['F_He_total']   = rho_He * Q_act / (M_He * 1e-3)
        p['F_H_total_in'] = p['F_He_total'] * p['p_H_total'] / (p['P_He'] - p['p_H_total'])

    p['F_He_tube'] = p['F_He_total'] / p['n_tubes']

    # Gas-phase mass transfer (H2 representative)
    mt = mass_transfer_coeff(
        p['T'], p['P_He'], p['Q_He_nlpm'],
        p['d_i'], p['n_tubes'], species='H2'
    )
    p['k_m']    = mt['k_m']
    p['Re']     = mt['Re']
    p['Sc']     = mt['Sc']
    p['Sh']     = mt['Sh']
    p['regime'] = mt['regime']

    # Feed-side resistances [m²·s·Pa/mol molecules]
    # R1 = RT/k_m  (gas film, molecular units: J_H2_mol = k_m/RT * Delta_p)
    # R2 = 1/K_a   (surface, molecular units from Vadrucci)
    p['R1'] = R_GAS * p['T'] / p['k_m']
    p['R2_H'] = surface_resistance_H(p['T'])
    p['R2_T'] = surface_resistance_T(p['T'], p['Ka_scaling'])

    # K_eff_f [mol molecules/m²/s/Pa] — combined feed-side coefficient
    # 1/K_eff_f = R1 + R2
    # Use H surface resistance as representative (R2_T ≈ R2_H for Ka_scaling='equal')
    p['K_eff_f'] = 1.0 / (p['R1'] + p['R2_H'])

    # Permeate-side K_a (assumed symmetric with feed side)
    p['K_a_s'] = 1.0 / p['R2_H']

    # Inlet molecular flows [mol molecules/s] per bundle
    # p_k = F_k_tube / (F_k_tube + F_He_tube) * P_He
    # -> F_k_tube = F_He_tube * p_k / (P_He - p_k)
    # Inlet molecular flows from F_H_total_in and composition
    p['F_H2_in'] = p['x_H_in']  * p['F_H_total_in']
    p['F_T2_in'] = p['x_T_in']  * p['F_H_total_in']

    return p


# ── Local thermodynamic quantities ────────────────────────────────────────────

def bulk_pressures(F_H2: float, F_T2: float, p: dict) -> tuple:
    """
    Bulk partial pressures and mole fractions from molecular bundle flows.
    Returns (p_H2, p_T2, P_H_total, x_H, x_T).
    """
    F_H2 = max(F_H2, 0.0)
    F_T2 = max(F_T2, 0.0)
    F_mol_tube = (F_H2 + F_T2) / p['n_tubes']
    F_He_tube  = p['F_He_tube']
    F_tot_tube = F_mol_tube + F_He_tube

    P_H_total = F_mol_tube / F_tot_tube * p['P_He']
    x_H = F_H2 / (F_H2 + F_T2) if (F_H2 + F_T2) > 0 else 0.5
    x_T = 1.0 - x_H

    p_H2 = x_H * P_H_total
    p_T2 = x_T * P_H_total

    return p_H2, p_T2, P_H_total, x_H, x_T


def phi_eff(x_H: float, x_T: float, p: dict) -> float:
    """Effective permeability [mol/m/s/Pa^0.5] at local composition."""
    return x_H * p['Phi_H2'] + x_T * p['Phi_T2']


# ── Implicit flux solver ──────────────────────────────────────────────────────

def solve_flux(P_H_total: float, x_H: float, x_T: float, p: dict) -> dict:
    """
    Solve the implicit molecular flux equation:

        J_mol = (Phi_eff/delta) * ( sqrt(P_H_total - J_mol/K_eff_f)
                                     - sqrt(J_mol/K_a_s + p_perm)   )

    Everything in molecular units [mol molecules/m²/s].

    Returns dict with J_mol, J_H2, J_T2, P_surf_feed, P_surf_perm, Phi_eff.
    """
    if P_H_total <= 0.0:
        return _zero_flux()

    Phi_e  = phi_eff(x_H, x_T, p)
    Keff_f = p['K_eff_f']
    Ka_s   = p['K_a_s']
    delta  = p['delta']
    p_perm = p['p_perm']
    coeff  = Phi_e / delta   # [mol/m2/s/Pa^0.5] — NO factor of 2

    J_max = Keff_f * max(P_H_total - p_perm, 0.0)
    if J_max <= 0.0:
        return _zero_flux()

    def F_val(J):
        af = P_H_total - J / Keff_f
        ap = p_perm    + J / Ka_s
        if af < 0:
            return J
        return J - coeff * (np.sqrt(af) - np.sqrt(ap))

    def dF_dJ(J):
        af = P_H_total - J / Keff_f
        ap = p_perm    + J / Ka_s
        if af <= 0 or ap <= 0:
            return 1.0
        return (1.0
                + coeff * 0.5 / (Keff_f * np.sqrt(af))
                + coeff * 0.5 / (Ka_s   * np.sqrt(ap)))

    # Initial guess: diffusion-limited estimate
    J = min(coeff * (np.sqrt(P_H_total) - np.sqrt(p_perm)), J_max * 0.99)
    J = max(J, 0.0)

    converged = False
    for _ in range(p['newton_maxiter']):
        fval = F_val(J)
        if abs(fval) < p['newton_tol']:
            converged = True
            break
        J = float(np.clip(J - fval / dF_dJ(J), 0.0, J_max))

    if not converged:
        try:
            f0, f1 = F_val(0.0), F_val(J_max)
            if f0 * f1 < 0:
                J = brentq(F_val, 0.0, J_max,
                           xtol=p['newton_tol'], rtol=1e-10, maxiter=200)
            else:
                J = 0.0 if abs(f0) < abs(f1) else J_max
        except Exception as e:
            warnings.warn(f"Flux solver failed: {e}")
            J = 0.0

    J = max(J, 0.0)

    # Split total molecular flux into species
    if Phi_e > 0:
        J_H2 = J * x_H * p['Phi_H2'] / Phi_e
        J_T2 = J * x_T * p['Phi_T2'] / Phi_e
    else:
        J_H2 = J_T2 = 0.0

    P_surf_feed = P_H_total - J / Keff_f
    P_surf_perm = p_perm    + J / Ka_s

    return {
        'J_mol':       J,
        'J_H2':        J_H2,
        'J_T2':        J_T2,
        'P_surf_feed': P_surf_feed,
        'P_surf_perm': P_surf_perm,
        'Phi_eff':     Phi_e,
        'K_eff_f':     Keff_f,
    }


def _zero_flux():
    return {k: 0.0 for k in
            ('J_mol', 'J_H2', 'J_T2', 'P_surf_feed', 'P_surf_perm',
             'Phi_eff', 'K_eff_f')}


# ── Regime diagnostic ─────────────────────────────────────────────────────────

def compute_lambda(P_H_total: float, x_H: float, x_T: float, p: dict) -> float:
    """
    Lambda = (Phi_eff/delta) / (K_eff_f * sqrt(P_H_total))
           = J_diff_lim / J_surf_lim    [dimensionless, molecular units]

    Lambda > 1: diffusion-limited flux > surface-limited flux
                -> surface/gas-film is the bottleneck -> surface-limited regime
    Lambda < 1: diffusion-limited flux < surface-limited flux
                -> bulk diffusion is the bottleneck -> diffusion-limited regime
    Lambda ~ 1: intermediate

    No factor of 2 here — consistent molecular units throughout.
    """
    if P_H_total <= 0:
        return np.inf
    Phi_e  = phi_eff(x_H, x_T, p)
    Keff_f = p['K_eff_f']
    return (Phi_e / p['delta']) / (Keff_f * np.sqrt(P_H_total))


def lambda_regime(lam: float) -> str:
    if lam > 100:    return 'surface-limited'
    elif lam > 0.01: return 'intermediate'
    else:            return 'diffusion-limited'


# ── ODE system ────────────────────────────────────────────────────────────────

def rhs(z: float, F: np.ndarray, p: dict) -> np.ndarray:
    """
    ODE RHS: d[F_H2, F_T2]/dz  [mol molecules/s/m]

    F : [F_H2, F_T2]  total bundle molecular flows [mol/s]
    """
    F_H2, F_T2 = max(F[0], 0.0), max(F[1], 0.0)
    _, _, P_H, x_H, x_T = bulk_pressures(F_H2, F_T2, p)
    fl = solve_flux(P_H, x_H, x_T, p)

    area = np.pi * p['d_i'] * p['n_tubes']
    return np.array([-fl['J_H2'] * area,
                     -fl['J_T2'] * area])


def jac(z: float, F: np.ndarray, p: dict) -> np.ndarray:
    """Finite-difference Jacobian for stiff solvers."""
    eps = max(1e-8 * np.max(np.abs(F)), 1e-16)
    J0  = rhs(z, F, p)
    jac_mat = np.zeros((2, 2))
    for i in range(2):
        Fp       = F.copy()
        Fp[i]   += eps
        jac_mat[:, i] = (rhs(z, Fp, p) - J0) / eps
    return jac_mat


# ── Main solver ───────────────────────────────────────────────────────────────

def solve_permeator(user_params: dict = None) -> dict:
    """
    Solve the permeator ODE and return full solution and diagnostics.

    Returns
    -------
    result dict:
        z            : axial positions [m]
        F_H2, F_T2   : molecular flows [mol/s]        shape (n_out,)
        P_H_total    : total H partial pressure [Pa]   shape (n_out,)
        x_H, x_T     : mole fractions                  shape (n_out,)
        J_mol        : total molecular flux [mol/m²/s] shape (n_out,)
        J_H2, J_T2   : species molecular flux          shape (n_out,)
        P_surf_feed  : feed surface pressure [Pa]      shape (n_out,)
        Lambda       : regime parameter                 shape (n_out,)
        eta_H2, eta_T2: recovery fractions             scalars
        diagnostics  : dict of transport properties
        params       : full params dict
        ivp_result   : raw scipy result
    """
    if user_params is None:
        user_params = {}
    p = build_params(user_params)

    F0     = np.array([p['F_H2_in'], p['F_T2_in']])
    z_eval = np.linspace(0.0, p['L'], p['n_out'])

    use_jac = p['solver'] in ('Radau', 'BDF', 'LSODA')
    sol = solve_ivp(
        fun     = lambda z, F: rhs(z, F, p),
        t_span  = (0.0, p['L']),
        y0      = F0,
        method  = p['solver'],
        t_eval  = z_eval,
        rtol    = p['rtol'],
        atol    = p['atol'],
        jac     = (lambda z, F: jac(z, F, p)) if use_jac else None,
    )

    if not sol.success:
        warnings.warn(f"solve_ivp: {sol.message}")

    F_out = np.maximum(sol.y, 0.0)
    n     = len(sol.t)

    # Recompute profiles
    P_H_out      = np.zeros(n)
    x_H_out      = np.zeros(n)
    x_T_out      = np.zeros(n)
    J_mol_out    = np.zeros(n)
    J_H2_out     = np.zeros(n)
    J_T2_out     = np.zeros(n)
    P_surf_out   = np.zeros(n)
    Lambda_out   = np.zeros(n)

    for i in range(n):
        _, _, P_H, x_H, x_T = bulk_pressures(F_out[0, i], F_out[1, i], p)
        fl = solve_flux(P_H, x_H, x_T, p)

        P_H_out[i]    = P_H
        x_H_out[i]    = x_H
        x_T_out[i]    = x_T
        J_mol_out[i]  = fl['J_mol']
        J_H2_out[i]   = fl['J_H2']
        J_T2_out[i]   = fl['J_T2']
        P_surf_out[i] = fl['P_surf_feed']
        Lambda_out[i] = compute_lambda(P_H, x_H, x_T, p)

    # Recovery fractions
    def eta(F_in, F_out_val):
        return (F_in - F_out_val) / F_in if F_in > 0 else 0.0

    eta_H2 = eta(F_out[0, 0], F_out[0, -1])
    eta_T2 = eta(F_out[1, 0], F_out[1, -1])

    diagnostics = {
        'Re':           p['Re'],
        'Sc':           p['Sc'],
        'Sh':           p['Sh'],
        'k_m':          p['k_m'],
        'flow_regime':  p['regime'],
        'R1_gas_film':  p['R1'],
        'R2_surface_H': p['R2_H'],
        'R2_surface_T': p['R2_T'],
        'K_eff_f':      p['K_eff_f'],
        'Phi_H2':       p['Phi_H2'],
        'Phi_T2':       p['Phi_T2'],
        'KS':           p['KS'],
        'A_total':      np.pi * p['d_i'] * p['L'] * p['n_tubes'],
        'Lambda_inlet': Lambda_out[0],
        'Lambda_exit':  Lambda_out[-1],
        'back_diffusion_H2': bool(np.any(J_H2_out < -1e-20)),
        'back_diffusion_T2': bool(np.any(J_T2_out < -1e-20)),
        'p_perm_fraction': {
            'H2': np.sqrt(p['p_perm']) / np.sqrt(max(p['p_H2_in'], 1e-30)),
            'T2': np.sqrt(p['p_perm']) / np.sqrt(max(p['p_T2_in'], 1e-30)),
        },
    }

    return {
        'z':           sol.t,
        'F_H2':        F_out[0],
        'F_T2':        F_out[1],
        'P_H_total':   P_H_out,
        'x_H':         x_H_out,
        'x_T':         x_T_out,
        'J_mol':       J_mol_out,
        'J_H2':        J_H2_out,
        'J_T2':        J_T2_out,
        'P_surf_feed': P_surf_out,
        'Lambda':      Lambda_out,
        'eta_H2':      eta_H2,
        'eta_T2':      eta_T2,
        # Keep N aliases for backward compatibility with plots.py
        'N':           np.array([F_out[0], F_out[1]]),
        'eta_H':       eta_H2,
        'eta_T':       eta_T2,
        'x_H':         x_H_out,
        'x_T':         x_T_out,
        'diagnostics': diagnostics,
        'params':      p,
        'ivp_result':  sol,
    }


# ── Result printing ───────────────────────────────────────────────────────────

def print_result(result: dict) -> None:
    p  = result['params']
    d  = result['diagnostics']
    eH = result['eta_H2']
    eT = result['eta_T2']
    La = result['Lambda']

    print(f"\n{'='*65}")
    print(f"Permeator — H2/T2 binary, molecular units")
    print(f"{'='*65}")
    print(f"\n  Geometry:")
    print(f"    d_i={p['d_i']*1e3:.1f} mm | delta={p['delta']*1e6:.0f} µm | "
          f"L={p['L']:.2f} m | n_tubes={p['n_tubes']}")
    print(f"    A_total = {d['A_total']*1e4:.1f} cm²")
    print(f"\n  Operating:")
    print(f"    T={p['T']:.0f} K | P_He={p['P_He']/1e5:.2f} bar | "
          f"Q_He={p['Q_He_nlpm']:.3f} Nl/min | p_perm={p['p_perm']:.1f} Pa")
    print(f"\n  Feed:")
    print(f"    F_H_total = {p['F_H_total_in']*1e6:.4f} µmol/s  "          f"({p['F_H_total_in']:.4e} mol/s)")
    print(f"    F_He      = {p['F_He_total']:.4e} mol/s")
    print(f"    P_H_total = {p['p_H_total']:.1f} Pa")
    print(f"    x_H = {p['x_H_in']:.3f}  F_H2 = {p['F_H2_in']*1e6:.4f} µmol/s")
    print(f"    x_T = {p['x_T_in']:.3f}  F_T2 = {p['F_T2_in']*1e6:.4f} µmol/s")
    print(f"\n  Transport ({d['flow_regime']}, Re={d['Re']:.0f}):")
    Rt = d['R1_gas_film'] + d['R2_surface_H']
    print(f"    R1 gas film : {d['R1_gas_film']:.3e}  ({100*d['R1_gas_film']/Rt:.1f}%)")
    print(f"    R2 surface  : {d['R2_surface_H']:.3e}  ({100*d['R2_surface_H']/Rt:.1f}%)")
    print(f"\n  Permeabilities:")
    print(f"    Phi_H2 = {d['Phi_H2']:.3e}  Phi_T2 = {d['Phi_T2']:.3e}")
    print(f"    Phi_H2/Phi_T2 = {d['Phi_H2']/d['Phi_T2']:.4f} "
          f"(expect sqrt(m_T/m_H) = {np.sqrt(M_T/M_H):.4f})")
    print(f"\n  Lambda (= Phi_eff/(delta*K_eff_f*sqrt(P_H)), molecular units):")
    print(f"    Inlet: {d['Lambda_inlet']:.3f}  ->  {lambda_regime(d['Lambda_inlet'])}")
    print(f"    Exit:  {d['Lambda_exit']:.3f}  ->  {lambda_regime(d['Lambda_exit'])}")
    print(f"\n  Recovery:")
    print(f"    eta_H2 = {100*eH:.3f}%")
    print(f"    eta_T2 = {100*eT:.3f}%")


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("Test 1: Default 50:50 H2/T2")
    r = solve_permeator()
    print_result(r)

    print("\n\nTest 2: Analytical limits")
    p_test = build_params({})
    P_H = 100.0; x_H = 0.5; x_T = 0.5

    # Diffusion-limited limit (K_eff_f -> inf)
    p_diff = {**p_test, 'K_eff_f': 1e10, 'K_a_s': 1e10}
    fl_diff = solve_flux(P_H, x_H, x_T, p_diff)
    Phi_e  = phi_eff(x_H, x_T, p_test)
    J_diff_expected = Phi_e / p_test['delta'] * (np.sqrt(P_H) - np.sqrt(p_test['p_perm']))
    print(f"\n  Diff-lim J_mol:   {fl_diff['J_mol']:.4e}")
    print(f"  Expected:          {J_diff_expected:.4e}")
    print(f"  Match: {abs(fl_diff['J_mol']-J_diff_expected)/J_diff_expected < 0.001}")

    # Surface-limited limit (Phi/delta -> inf)
    p_surf = {**p_test}
    p_surf['Phi_H2'] = 1e10 * p_test['Phi_H2']
    p_surf['Phi_T2'] = 1e10 * p_test['Phi_T2']
    fl_surf = solve_flux(P_H, x_H, x_T, p_surf)
    J_surf_expected = p_test['K_eff_f'] * (P_H - p_test['p_perm'])
    print(f"\n  Surf-lim J_mol:   {fl_surf['J_mol']:.4e}")
    print(f"  Expected:          {J_surf_expected:.4e}")
    print(f"  Match: {abs(fl_surf['J_mol']-J_surf_expected)/J_surf_expected < 0.001}")

    print("\n\nTest 3: L->0 gives zero recovery")
    r = solve_permeator({'L': 1e-6})
    print(f"  eta_H2 = {100*r['eta_H2']:.6f}%  eta_T2 = {100*r['eta_T2']:.6f}%")

    print("\n\nTest 4: Temperature sensitivity")
    print(f"{'T[K]':>6}  {'eta_H2[%]':>10}  {'eta_T2[%]':>10}  {'Lambda_in':>12}  {'Regime':>20}")
    for T in [473, 523, 573, 623, 673, 723]:
        r = solve_permeator({'T': T})
        lam = r['diagnostics']['Lambda_inlet']
        print(f"{T:6.0f}  {100*r['eta_H2']:10.3f}  {100*r['eta_T2']:10.3f}  "
              f"{lam:12.4f}  {lambda_regime(lam):>20}")

    print("\n\nTest 5: Lambda units check")
    p_c = build_params({})
    P_H = 100.0; x_H = 0.5; x_T = 0.5
    Phi_e = phi_eff(x_H, x_T, p_c)
    Keff  = p_c['K_eff_f']
    delta = p_c['delta']
    lam   = (Phi_e/delta) / (Keff * np.sqrt(P_H))
    J_diff_lim = Phi_e/delta * np.sqrt(P_H)
    J_surf_lim = Keff * P_H
    print(f"  Phi_eff/delta = {Phi_e/delta:.3e} mol/m2/s/Pa^0.5")
    print(f"  K_eff_f       = {Keff:.3e} mol/m2/s/Pa")
    print(f"  sqrt(P_H)     = {np.sqrt(P_H):.3f} Pa^0.5")
    print(f"  Lambda        = {lam:.4f}  (= {J_diff_lim:.3e} / {J_surf_lim:.3e})")
    print(f"  = J_diff_lim [mol H2/m2/s] / J_surf_lim [mol H2/m2/s]")
    print(f"  Units: [mol/m2/s/Pa^0.5] / ([mol/m2/s/Pa] * [Pa^0.5]) = dimensionless ✓")