"""
transport.py
============
Gas-phase and surface transport properties for the He purge loop permeator.

Covers:
  - Binary diffusivity D_HT-He(T, P)  [Chung & Dalgarno 2002]
  - He viscosity and density
  - Reynolds, Schmidt, Sherwood numbers
  - Mass transfer coefficient k_m
  - Surface mass transfer coefficient K_a  [Vadrucci 2013 / Antunes 2020]
  - Combined feed-side resistance K_eff_f
  - Regime diagnostic Lambda(z)
"""

import numpy as np
import warnings
from material_library import (
    M_H, M_D, M_T, M_He, M_H2, M_HT, M_T2,
    sieverts_constant, permeability_species
)

R_GAS     = 8.314        # J/mol/K
R_GAS_kJ  = 8.314e-3     # kJ/mol/K


# ── Gas-phase binary diffusivity ──────────────────────────────────────────────

def diffusivity_HT_He(T: float, P: float) -> float:
    """
    Binary diffusivity of HT in He [m²/s].

    Base: Chung & Dalgarno (2002) fit for atomic H in He at 1 atm:
        D_H-He(T, P_ref) = 1.032e-8 * T^1.74   [m²/s]
    Valid for T > 273 K. Fit error < 2%.

    Corrections applied:
    1. Pressure scaling: D ∝ 1/P  (Chapman-Enskog, ideal gas)
    2. Molecular mass correction: H atom → HT molecule in He
       Uses reduced-mass formula from Chapman-Enskog theory:
       D ∝ sqrt(1/μ_reduced) where μ = m1*m2/(m1+m2)

    Parameters
    ----------
    T : temperature [K]
    P : total pressure [Pa]

    Returns
    -------
    D_HT_He [m²/s]
    """
    if T < 273:
        warnings.warn(
            f"T = {T} K is below the Chung & Dalgarno fit range (273 K). "
            "Extrapolation may be unreliable.",
            stacklevel=2
        )

    P_ref = 101325.0  # Pa

    # Atomic H in He at P_ref
    D_H_He_ref = 1.032e-8 * T ** 1.74   # m²/s

    # Pressure scaling
    D_H_He = D_H_He_ref * P_ref / P

    # Reduced masses
    mu_H_He  = M_H  * M_He / (M_H  + M_He)   # H atom in He
    mu_HT_He = M_HT * M_He / (M_HT + M_He)   # HT molecule in He

    # Chapman-Enskog: D ∝ sqrt(1/mu_reduced) at same collision integral
    # This is an approximation — collision integrals differ for H and HT
    # but at these conditions the error is ~5-10%
    mass_correction = np.sqrt(mu_H_He / mu_HT_He)

    D_HT_He = D_H_He * mass_correction

    return D_HT_He


def diffusivity_H2_He(T: float, P: float) -> float:
    """Binary diffusivity of H₂ in He [m²/s]. Same correction approach."""
    P_ref  = 101325.0
    D_ref  = 1.032e-8 * T ** 1.74
    D_H_He = D_ref * P_ref / P
    mu_H_He  = M_H  * M_He / (M_H  + M_He)
    mu_H2_He = M_H2 * M_He / (M_H2 + M_He)
    return D_H_He * np.sqrt(mu_H_He / mu_H2_He)


def diffusivity_T2_He(T: float, P: float) -> float:
    """Binary diffusivity of T₂ in He [m²/s]. Same correction approach."""
    P_ref  = 101325.0
    D_ref  = 1.032e-8 * T ** 1.74
    D_H_He = D_ref * P_ref / P
    mu_H_He  = M_H  * M_He / (M_H  + M_He)
    mu_T2_He = M_T2 * M_He / (M_T2 + M_He)
    return D_H_He * np.sqrt(mu_H_He / mu_T2_He)


def species_diffusivity_He(T: float, P: float, species: str) -> float:
    """
    Binary diffusivity of hydrogen isotopologue in He [m²/s].

    Parameters
    ----------
    species : 'H2', 'HT', or 'T2'
    """
    funcs = {
        'H2': diffusivity_H2_He,
        'HT': diffusivity_HT_He,
        'T2': diffusivity_T2_He,
    }
    if species not in funcs:
        raise ValueError(f"species must be 'H2', 'HT', or 'T2', got '{species}'")
    return funcs[species](T, P)


# ── He carrier gas properties ─────────────────────────────────────────────────

def viscosity_He(T: float) -> float:
    """
    Dynamic viscosity of He [Pa·s].

    Power-law fit: mu = mu_ref * (T/T_ref)^0.67
    mu_ref = 1.99e-5 Pa·s at T_ref = 300 K.
    """
    return 1.99e-5 * (T / 300.0) ** 0.67


def density_He(T: float, P: float) -> float:
    """
    Density of He carrier gas [kg/m³].

    Ideal gas: rho = P * M_He / (R * T)
    """
    return P * (M_He * 1e-3) / (R_GAS * T)


# ── Reynolds, Schmidt, Sherwood numbers ──────────────────────────────────────

def reynolds_number(
    T: float, P: float, Q_total_nlpm: float, d_i: float, n_tubes: int
) -> dict:
    """
    Reynolds number per tube.

    Parameters
    ----------
    T              : temperature [K]
    P              : total pressure [Pa]
    Q_total_nlpm   : total He volumetric flow [Nl/min] (normal litres/min)
    d_i            : tube inner diameter [m]
    n_tubes        : number of parallel tubes

    Returns
    -------
    dict with Re, u_bar, mdot_tube, rho, mu
    """
    # Convert normal litres/min to actual m³/s per tube
    # Normal conditions: T_n = 273.15 K, P_n = 101325 Pa
    T_n = 273.15
    P_n = 101325.0
    Q_actual_total = Q_total_nlpm / 60.0 * 1e-3 * (T / T_n) * (P_n / P)  # m³/s
    Q_tube = Q_actual_total / n_tubes  # m³/s per tube

    rho   = density_He(T, P)
    mu    = viscosity_He(T)
    A_cs  = np.pi * (d_i / 2) ** 2    # cross-sectional area [m²]
    u_bar = Q_tube / A_cs              # mean velocity [m/s]
    Re    = rho * u_bar * d_i / mu

    return {
        'Re':         Re,
        'u_bar':      u_bar,       # m/s
        'Q_tube':     Q_tube,      # m³/s
        'rho':        rho,         # kg/m³
        'mu':         mu,          # Pa·s
    }


def schmidt_number(
    T: float, P: float, species: str = 'HT'
) -> float:
    """
    Schmidt number for hydrogen isotopologue in He.

    Sc = mu / (rho * D_species_He)
    """
    mu  = viscosity_He(T)
    rho = density_He(T, P)
    D   = species_diffusivity_He(T, P, species)
    return mu / (rho * D)


def sherwood_number(Re: float, Sc: float) -> tuple[float, str]:
    """
    Sherwood number for flow in a tube.

    Correlations:
      Laminar   (Re < 2300)  : Sh = 3.66
        Graetz solution, fully developed, constant wall concentration BC.
      Turbulent (Re > 10000) : Sh = 0.023 * Re^0.8 * Sc^0.33
        Dittus-Boelter (heating).
      Transition (2300–10000): linear interpolation — flagged as uncertain.

    Returns
    -------
    (Sh, regime_label)
    """
    if Re < 2300:
        return 3.66, 'laminar'
    elif Re > 10000:
        Sh_turb = 0.023 * Re ** 0.8 * Sc ** 0.33
        return Sh_turb, 'turbulent'
    else:
        Sh_lam  = 3.66
        Sh_turb = 0.023 * Re ** 0.8 * Sc ** 0.33
        f = (Re - 2300) / (10000 - 2300)
        Sh = Sh_lam + f * (Sh_turb - Sh_lam)
        return Sh, 'transition (interpolated — uncertain)'


def mass_transfer_coeff(
    T: float, P: float, Q_total_nlpm: float,
    d_i: float, n_tubes: int, species: str = 'HT'
) -> dict:
    """
    Gas-phase mass transfer coefficient k_m [m/s] and associated diagnostics.

    k_m = Sh * D_species_He / d_i

    Returns
    -------
    dict with k_m, Re, Sc, Sh, regime, D_species_He, and gas-film resistance RT/k_m
    """
    re_dict = reynolds_number(T, P, Q_total_nlpm, d_i, n_tubes)
    Re  = re_dict['Re']
    Sc  = schmidt_number(T, P, species)
    Sh, regime = sherwood_number(Re, Sc)
    D   = species_diffusivity_He(T, P, species)
    k_m = Sh * D / d_i

    # Gas-film resistance in p-space [m²·s·Pa/mol]
    # Note: RT/k_m NOT 1/(k_m * K_S) — the gas phase is linear (Henry's law)
    R_gas_film = R_GAS * T / k_m

    return {
        'k_m':          k_m,            # m/s
        'Re':           Re,
        'Sc':           Sc,
        'Sh':           Sh,
        'regime':       regime,
        'D_species_He': D,              # m²/s
        'R_gas_film':   R_gas_film,     # m²·s·Pa/mol  (p-space resistance)
        'u_bar':        re_dict['u_bar'],
        'rho':          re_dict['rho'],
        'mu':           re_dict['mu'],
    }


# ── Surface mass transfer coefficient K_a ────────────────────────────────────

def surface_resistance_H(T: float) -> float:
    """
    Surface mass transfer resistance for H on Pd/Ag [m²·s·Pa/mol].

    From Vadrucci et al. (2013), cited as Eq. (5) in Antunes et al. (2020):
        R_a,H = 1/K_a,H = 488 * exp(+20103 / (R_J * T))

    where 20103 has units of J/mol, so E_a = 20.1 kJ/mol.
    Consistent with clean Pd/Ag and Pick & Sonnenberg (1985) range 15-25 kJ/mol.

    Note: positive exponent — surface resistance worsens on cooling.
    At 623 K: R_a,H ≈ 2.4e4 m²·s·Pa/mol  →  K_a ≈ 4.2e-5 mol/m²/s/Pa
    At 573 K: R_a,H ≈ 3.3e4 m²·s·Pa/mol  →  K_a ≈ 3.0e-5 mol/m²/s/Pa

    Parameters
    ----------
    T : membrane temperature [K]

    Returns
    -------
    R_a_H [m²·s·Pa/mol]  — reciprocal of K_a,H
    """
    # R_GAS = 8.314 J/mol/K; 20103 is in J/mol giving E_a = 20.1 kJ/mol
    return 488.0 * np.exp(20103.0 / (R_GAS * T))


def surface_resistance_T(T: float, scaling: str = 'equal') -> float:
    """
    Surface mass transfer resistance for T on Pd/Ag [m²·s·Pa/mol].

    No measured K_a^T data in literature. Two options:

    'equal'  : R_a,T = R_a,H  (default — conservative assumption)
    'scaled' : R_a,T = R_a,H * sqrt(m_T/m_H)
               (classical rate theory: adsorption frequency ∝ 1/sqrt(m))

    Parameters
    ----------
    scaling : 'equal' or 'scaled'

    Returns
    -------
    R_a_T [m²·s·Pa/mol]
    """
    R_aH = surface_resistance_H(T)
    if scaling == 'equal':
        return R_aH
    elif scaling == 'scaled':
        return R_aH * np.sqrt(M_T / M_H)
    else:
        raise ValueError(f"scaling must be 'equal' or 'scaled', got '{scaling}'")


def Ka_species(T: float, species: str, scaling: str = 'equal') -> float:
    """
    Surface mass transfer coefficient K_a for a molecular species [mol/m²/s/Pa].

    Maps atomic K_a to molecular species:
      H₂ → K_a^H
      HT → geometric mean of K_a^H and K_a^T
      T₂ → K_a^T

    Parameters
    ----------
    species : 'H2', 'HT', or 'T2'
    scaling : isotope scaling for T ('equal' or 'scaled')

    Returns
    -------
    K_a [mol/m²/s/Pa]
    """
    Ka_H = 1.0 / surface_resistance_H(T)
    Ka_T = 1.0 / surface_resistance_T(T, scaling)

    if species == 'H2':
        return Ka_H
    elif species == 'T2':
        return Ka_T
    elif species == 'HT':
        # Geometric mean — same logic as permeability isotope scaling
        return np.sqrt(Ka_H * Ka_T)
    else:
        raise ValueError(f"species must be 'H2', 'HT', or 'T2', got '{species}'")


# ── Combined feed-side resistance K_eff_f ────────────────────────────────────

def K_eff_f(
    T: float, P: float, Q_total_nlpm: float,
    d_i: float, n_tubes: int,
    species: str, Ka_scaling: str = 'equal'
) -> dict:
    """
    Combined feed-side resistance coefficient K_eff^f [mol/m²/s/Pa].

    1/K_eff^f = RT/k_m  +  1/K_a^f

    Both terms operate in p-space (linear driving force).

    Also returns the relative contribution of each term.

    Returns
    -------
    dict with K_eff_f, R_gas_film, R_surface, f_surface (surface fraction)
    """
    mt = mass_transfer_coeff(T, P, Q_total_nlpm, d_i, n_tubes, species)
    R_film    = mt['R_gas_film']                          # RT/k_m [m²·s·Pa/mol]
    R_surface = surface_resistance_H(T)                  # 1/K_a^f [m²·s·Pa/mol]
    # Use H resistance for all species by default (K_a^T unknown)
    # Could use Ka_species for species-specific, but adds uncertainty
    R_total   = R_film + R_surface
    K_eff     = 1.0 / R_total

    return {
        'K_eff_f':    K_eff,        # mol/m²/s/Pa
        'R_gas_film': R_film,       # m²·s·Pa/mol
        'R_surface':  R_surface,    # m²·s·Pa/mol
        'R_total_f':  R_total,      # m²·s·Pa/mol
        'f_surface':  R_surface / R_total,   # fraction of resistance in surface step
        'f_gas_film': R_film / R_total,      # fraction in gas-film step (expect ~0)
        'k_m':        mt['k_m'],
        'Re':         mt['Re'],
        'Sc':         mt['Sc'],
        'Sh':         mt['Sh'],
        'regime':     mt['regime'],
    }


# ── Regime diagnostic Lambda(z) ───────────────────────────────────────────────

def Lambda(
    mat: dict, T: float, p_bulk: float,
    delta: float, f_phi: float = 1.0,
    species: str = 'HT', Ka_scaling: str = 'equal'
) -> float:
    """
    Regime diagnostic parameter Λ.

    Λ = 2Φ_k / δ / (K_eff^f × √P_H,total)

    The factor 2 comes from the governing equation using 2*Phi_eff/delta
    as the diffusion coefficient (K_S is in mol(H2)/m3/Pa^0.5 so each
    molecule gives 2 atoms — the factor 2 is needed for consistency).

    Λ ≫ 1 : surface-limited  (your operating regime)
    Λ ≈ 1 : intermediate
    Λ ≪ 1 : diffusion-limited

    Note: Λ depends on p_bulk(z), so it varies along the tube.
    As HT is extracted, p_bulk falls, Λ rises — regime deepens toward
    surface-limited toward the bleed end.

    Parameters
    ----------
    mat     : material dict
    T       : temperature [K]
    p_bulk  : local bulk partial pressure of species [Pa]
    delta   : membrane thickness [m]
    f_phi   : fouling factor
    species : 'H2', 'HT', or 'T2'

    Returns
    -------
    Λ (dimensionless)
    """
    if p_bulk <= 0:
        return np.inf

    Phi  = permeability_species(mat, T, species, f_phi)
    Ka   = Ka_species(T, species, Ka_scaling)
    return (Phi / delta) / (Ka * np.sqrt(p_bulk))


def Lambda_regime(lam: float) -> str:
    """Return regime label for a given Λ value."""
    if lam > 100:
        return 'surface-limited'
    elif lam > 0.01:
        return 'intermediate'
    else:
        return 'diffusion-limited'


# ── Diagnostic summary ────────────────────────────────────────────────────────

def print_transport_summary(
    T: float, P: float, Q_total_nlpm: float,
    d_i: float, n_tubes: int,
    mat: dict, delta: float, p_H_total: float,
    f_phi: float = 1.0, Ka_scaling: str = 'equal'
) -> None:
    """Print a formatted transport diagnostics summary."""

    print(f"\n{'='*65}")
    print(f"Transport diagnostics  |  T={T} K  P={P/1e5:.2f} bar  "
          f"Q={Q_total_nlpm} Nl/min")
    print(f"{'='*65}")

    # Gas-phase properties
    D_HT = diffusivity_HT_He(T, P)
    mu   = viscosity_He(T)
    rho  = density_He(T, P)
    mt   = mass_transfer_coeff(T, P, Q_total_nlpm, d_i, n_tubes, 'HT')

    print(f"\n  Gas-phase:")
    print(f"    D_HT-He       = {D_HT:.3e}  m²/s")
    print(f"    mu_He         = {mu:.3e}  Pa·s")
    print(f"    rho_He        = {rho:.3f}  kg/m³")
    print(f"    u_bar/tube    = {mt['u_bar']:.3f}  m/s")
    print(f"    Re            = {mt['Re']:.1f}  ({mt['regime']})")
    print(f"    Sc            = {mt['Sc']:.2f}")
    print(f"    Sh            = {mt['Sh']:.3f}")
    print(f"    k_m           = {mt['k_m']:.3e}  m/s")

    print(f"\n  Resistance breakdown (feed side, HT):")
    keff = K_eff_f(T, P, Q_total_nlpm, d_i, n_tubes, 'HT', Ka_scaling)
    print(f"    RT/k_m        = {keff['R_gas_film']:.3e}  m²·s·Pa/mol  "
          f"({100*keff['f_gas_film']:.4f}% of total)")
    print(f"    1/K_a^f       = {keff['R_surface']:.3e}  m²·s·Pa/mol  "
          f"({100*keff['f_surface']:.4f}% of total)")
    print(f"    Gas-film is {'NEGLIGIBLE' if keff['f_gas_film'] < 0.001 else 'SIGNIFICANT'}")

    print(f"\n  Regime diagnostic Λ (at inlet):")
    for sp in ('H2', 'HT', 'T2'):
        p_sp = p_H_total * {'H2': 0.05, 'HT': 0.90, 'T2': 0.05}[sp]
        lam  = Lambda(mat, T, p_sp, delta, f_phi, sp, Ka_scaling)
        label = Lambda_regime(lam)
        print(f"    Λ_{sp:3s}         = {lam:.2e}  → {label}")

    R_surf = surface_resistance_H(T)
    print(f"\n  Surface kinetics:")
    print(f"    1/K_a^H       = {R_surf:.3e}  m²·s·Pa/mol")
    print(f"    K_a^H         = {1/R_surf:.3e}  mol/m²/s/Pa")
    print(f"    Activation E  = 20.1 kJ/mol  (from Vadrucci 2013, 20103 J/mol / R)")
    print(f"    K_a isotope   : {Ka_scaling}")


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from material_library import get_material

    mat = get_material('Pd25Ag')

    # HELIX default operating conditions
    T           = 623.0    # K
    P           = 2e5      # Pa
    Q_nlpm      = 20.0     # Nl/min
    d_i         = 0.010    # m
    n_tubes     = 10
    delta       = 100e-6   # m
    p_H_total   = 100.0    # Pa

    print_transport_summary(
        T, P, Q_nlpm, d_i, n_tubes,
        mat, delta, p_H_total
    )

    # Show temperature sensitivity of K_a
    print("\n\nK_a temperature sensitivity (Pd/Ag, H):")
    print(f"{'T [K]':>8}  {'T [°C]':>8}  {'1/K_a [m²·s·Pa/mol]':>22}  {'K_a [mol/m²/s/Pa]':>20}")
    for T_test in [473, 523, 573, 623, 673, 723]:
        Ra = surface_resistance_H(T_test)
        print(f"{T_test:8.0f}  {T_test-273.15:8.1f}  {Ra:22.3e}  {1/Ra:20.3e}")

    # Show Lambda sensitivity to p_HT
    print("\n\nΛ vs p_HT (HT, Pd-25Ag, T=623 K, δ=100 µm):")
    print(f"{'p_HT [Pa]':>12}  {'Λ':>12}  {'Regime':>20}")
    for p_test in [1, 5, 10, 50, 100, 500, 1000, 1e4, 1e5]:
        lam = Lambda(mat, 623.0, p_test, 100e-6, species='HT')
        print(f"{p_test:12.1f}  {lam:12.2e}  {Lambda_regime(lam):>20}")
