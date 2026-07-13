"""
material_library.py
===================
Arrhenius material property database for the He purge loop Pd/Ag permeator model.

All parameters from Shimada (2020) Comprehensive Nuclear Materials 2nd ed., Tables 1 & 2,
unless otherwise noted.

Units throughout:
    D0      [m²/s]
    ED      [kJ/mol]
    KS0     [mol(H₂)/m³/Pa⁰·⁵]
    ES      [kJ/mol]
    Phi0    [mol(H₂)/m/s/Pa⁰·⁵]
    EPhi    [kJ/mol]
    T_min, T_max  [K]  — valid temperature range from experiments

Isotope scaling (classical rate theory, Shimada Table 1 header):
    D^H : D^D : D^T = 1 : 1/√2 : 1/√3
    K_S assumed isotope-independent
    Φ_Q = (1/√m_Q) × Φ_H where m_Q is atomic mass [amu]
"""

import numpy as np

R_GAS = 8.314e-3  # kJ/mol/K

# Atomic masses [amu]
M_H  = 1.008
M_D  = 2.014
M_T  = 3.016
M_He = 4.003

# Molecular masses [amu]
M_H2 = 2 * M_H
M_HT = M_H + M_T
M_T2 = 2 * M_T
M_D2 = 2 * M_D

# ── Material database ─────────────────────────────────────────────────────────

MATERIALS = {

    'Pd25Ag': {
        # Serra et al. (1998) Metall. Mater. Trans. A 29A, 1023–1028
        # Shimada (2020) Table 2
        # PRIMARY membrane material for HELIX/HCPB permeator
        'D0':    3.07e-7,   'ED':   25.9,   # diffusivity
        'KS0':   1.82e-1,   'ES':  -19.6,   # solubility — NOTE: ES negative
        'Phi0':  5.58e-8,   'EPhi':  6.3,   # permeability
        'T_min': 323,       'T_max': 773,
        'note': (
            'Primary HELIX membrane material. ES = -19.6 kJ/mol is negative — '
            'solubility increases with decreasing T. Valid 323–773 K. '
            'Permeability from Serra et al. (1998); Antunes uses slightly '
            'different fit (Phi0=2.8e-8, EPhi=33.4) from ENEA experiments.'
        ),
    },

    'Pd_pure': {
        # Volkl & Alefeld (1975) for D; Favreau et al. (1954) for KS
        # Shimada (2020) Table 1
        'D0':    2.90e-7,   'ED':   22.2,
        'KS0':   4.45e-1,   'ES':   -8.4,
        'Phi0':  1.29e-7,   'EPhi': 13.8,
        'T_min': 223,       'T_max': 873,
        'note': (
            'Pure Pd baseline. Higher permeability than Pd-25Ag at low T '
            'due to lower EPhi. D from 23-author consensus (Volkl & Alefeld).'
        ),
    },

    'Nb': {
        # Volkl & Alefeld (1975) for D; Veleckis & Edwards (1969) for KS
        # Shimada (2020) Table 1
        # Used as substrate for Pd-coated Nb composite membranes
        'D0':    5.00e-8,   'ED':   10.2,
        'KS0':   1.26e-1,   'ES':  -35.3,
        'Phi0':  6.30e-9,   'EPhi': -25.1,
        'T_min': 223,       'T_max': 873,
        'note': (
            'Group V metal. Very high bulk diffusivity but SURFACE SENSITIVE — '
            'oxygen contamination makes permeation measurements unreliable '
            '(Shimada §6.08.3.12). EPhi negative — permeability increases on '
            'cooling. Use bulk properties with caution for permeator design; '
            'surface kinetics likely rate-limiting even more than Pd/Ag.'
        ),
    },

    'V': {
        # Volkl & Alefeld (1975) for D; Veleckis & Edwards (1969) for KS
        # Shimada (2020) Table 1
        'D0':    2.90e-8,   'ED':    4.2,
        'KS0':   1.38e-1,   'ES':  -29.0,
        'Phi0':  4.00e-9,   'EPhi': -24.9,
        'T_min': 173,       'T_max': 827,
        'note': (
            'Very high diffusivity but highly surface-sensitive. '
            'Not recommended as standalone membrane material.'
        ),
    },

    'RAFM': {
        # Causey et al. (2012) Comprehensive Nuclear Materials, Shimada Table 2
        # For parasitic permeation through structural tubing, NOT as membrane
        'D0':    1.00e-7,   'ED':   13.2,
        'KS0':   4.40e-1,   'ES':   28.6,
        'Phi0':  4.40e-8,   'EPhi': 41.8,
        'T_min': 300,       'T_max': 973,
        'note': (
            'Structural material (F82H, Eurofer). Use only above 573 K '
            '(trapping negligible above this T). Not for membrane use — '
            'for calculating parasitic T permeation through structural wall.'
        ),
    },

    'SS316L': {
        # Reiter (1993) EUR 15217; Shimada (2020) Table 2
        # HELIX loop tubing material
        'D0':    8.70e-7,   'ED':   51.9,
        'KS0':   3.60e-1,   'ES':   11.7,
        'Phi0':  3.13e-7,   'EPhi': 63.6,
        'T_min': 500,       'T_max': 1200,
        'note': (
            'HELIX loop tubing. High EPhi means Phi drops sharply below 500 K — '
            'good tritium barrier at ambient T. Recommended values from '
            '14-study consensus (Reiter 1993).'
        ),
    },

}


# ── Property functions ────────────────────────────────────────────────────────

def get_material(name: str) -> dict:
    """Return material parameter dict, with validation."""
    if name not in MATERIALS:
        raise ValueError(
            f"Material '{name}' not in library. "
            f"Available: {list(MATERIALS.keys())}"
        )
    return MATERIALS[name]


def diffusivity(mat: dict, T: float) -> float:
    """
    Lattice diffusivity of H in material [m²/s].

    D_H(T) = D0 * exp(-ED / (R*T))

    Parameters
    ----------
    mat : material dict from MATERIALS
    T   : temperature [K]

    Returns
    -------
    D_H [m²/s]
    """
    _check_T_range(mat, T)
    return mat['D0'] * np.exp(-mat['ED'] / (R_GAS * T))


def diffusivity_isotope(mat: dict, T: float, isotope: str) -> float:
    """
    Diffusivity for a specific hydrogen isotope [m²/s].

    Classical rate theory: D^Q = D^H / sqrt(m_Q / m_H)

    Parameters
    ----------
    isotope : 'H', 'D', or 'T'
    """
    D_H = diffusivity(mat, T)
    mass_ratio = {'H': 1.0, 'D': M_D / M_H, 'T': M_T / M_H}
    if isotope not in mass_ratio:
        raise ValueError(f"isotope must be 'H', 'D', or 'T', got '{isotope}'")
    return D_H / np.sqrt(mass_ratio[isotope])


def sieverts_constant(mat: dict, T: float) -> float:
    """
    Sieverts constant K_S(T) [mol(H₂)/m³/Pa⁰·⁵].

    K_S(T) = KS0 * exp(-ES / (R*T))

    Assumed isotope-independent for Pd/Ag.
    Note: ES can be negative (solubility increases on cooling, as for Pd-25Ag).
    """
    _check_T_range(mat, T)
    return mat['KS0'] * np.exp(-mat['ES'] / (R_GAS * T))


def permeability(mat: dict, T: float, f_phi: float = 1.0) -> float:
    """
    Permeability of H₂ in material [mol(H₂)/m/s/Pa⁰·⁵].

    Phi_H(T) = (Phi0 / sqrt(m_H)) * exp(-EPhi / (R*T)) * f_phi

    f_phi : fouling factor [0, 1]. Default 1.0 (clean surface).
    """
    _check_T_range(mat, T)
    Phi_H = mat['Phi0'] / np.sqrt(M_H) * np.exp(-mat['EPhi'] / (R_GAS * T))
    return Phi_H * f_phi


def permeability_species(
    mat: dict, T: float, species: str, f_phi: float = 1.0
) -> float:
    """
    Permeability for a specific molecular hydrogen isotopologue [mol/m/s/Pa⁰·⁵].

    Scaling from Shimada (2020) Table 1 header:
        Phi_Q = (1/sqrt(m_Q_atomic)) * Phi0 * exp(-EPhi/RT)

    For molecular species the relevant atomic mass is:
        H₂  → m_H  (both atoms H)
        HT  → geometric mean sqrt(m_H * m_T)  → (m_H * m_T)^(1/4) denominator
        T₂  → m_T  (both atoms T)

    Parameters
    ----------
    species : 'H2', 'HT', or 'T2'
    f_phi   : fouling factor [0, 1]

    Returns
    -------
    Phi_species [mol(H₂)/m/s/Pa⁰·⁵]
    """
    _check_T_range(mat, T)
    base = mat['Phi0'] * np.exp(-mat['EPhi'] / (R_GAS * T))

    if species == 'H2':
        scale = np.sqrt(M_H)
    elif species == 'HT':
        # geometric mean of atomic masses → (m_H * m_T)^(1/4) in denominator
        scale = (M_H * M_T) ** 0.25
    elif species == 'T2':
        scale = np.sqrt(M_T)
    else:
        raise ValueError(f"species must be 'H2', 'HT', or 'T2', got '{species}'")

    return (base / scale) * f_phi


def permeability_all_species(
    mat: dict, T: float, f_phi: float = 1.0
) -> dict:
    """
    Return permeabilities for all three species as a dict.

    Returns
    -------
    {'H2': Phi_H2, 'HT': Phi_HT, 'T2': Phi_T2}  [mol/m/s/Pa⁰·⁵]
    """
    return {
        sp: permeability_species(mat, T, sp, f_phi)
        for sp in ('H2', 'HT', 'T2')
    }


# ── Consistency check ─────────────────────────────────────────────────────────

def check_phi_consistency(mat: dict, T: float) -> dict:
    """
    Verify Phi = D * K_S at temperature T.

    Returns dict with computed and stored values and relative discrepancy.
    Significant discrepancy (>10%) indicates the Arrhenius fits are from
    different datasets and should be used carefully.
    """
    D_H   = diffusivity(mat, T)
    K_S   = sieverts_constant(mat, T)
    Phi_H = permeability(mat, T)
    Phi_DKS = D_H * K_S

    rel_err = abs(Phi_H - Phi_DKS) / max(Phi_H, Phi_DKS)

    return {
        'T': T,
        'Phi_stored': Phi_H,
        'Phi_from_DKS': Phi_DKS,
        'D': D_H,
        'KS': K_S,
        'relative_discrepancy': rel_err,
        'consistent': rel_err < 0.10,
    }


def print_material_summary(name: str, T: float) -> None:
    """Print a formatted summary of material properties at temperature T."""
    mat = get_material(name)
    D   = diffusivity(mat, T)
    KS  = sieverts_constant(mat, T)
    phis = permeability_all_species(mat, T)
    chk = check_phi_consistency(mat, T)

    print(f"\n{'='*60}")
    print(f"Material: {name}  |  T = {T} K  ({T-273.15:.0f} °C)")
    print(f"{'='*60}")
    print(f"  D_H          = {D:.3e}  m²/s")
    print(f"  K_S          = {KS:.3e}  mol(H₂)/m³/Pa⁰·⁵")
    print(f"  Phi_H2       = {phis['H2']:.3e}  mol/m/s/Pa⁰·⁵")
    print(f"  Phi_HT       = {phis['HT']:.3e}  mol/m/s/Pa⁰·⁵")
    print(f"  Phi_T2       = {phis['T2']:.3e}  mol/m/s/Pa⁰·⁵")
    print(f"  Phi_H2/Phi_T2 = {phis['H2']/phis['T2']:.3f}  (isotope fractionation)")
    print(f"  Phi vs D*KS  : rel. discrepancy = {chk['relative_discrepancy']:.1%}"
          + ("  ✓" if chk['consistent'] else "  ⚠ >10%"))
    print(f"  T range      : {mat['T_min']}–{mat['T_max']} K")
    print(f"  Note         : {mat['note'][:80]}...")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _check_T_range(mat: dict, T: float) -> None:
    """Warn if T is outside the experimental validity range."""
    if not (mat['T_min'] <= T <= mat['T_max']):
        import warnings
        warnings.warn(
            f"T = {T} K is outside the experimental validity range "
            f"{mat['T_min']}–{mat['T_max']} K for this material. "
            "Arrhenius extrapolation may be unreliable.",
            stacklevel=3
        )


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Material library self-test")
    print("="*60)

    for name in MATERIALS:
        mat = get_material(name)
        T_mid = (mat['T_min'] + mat['T_max']) / 2
        try:
            print_material_summary(name, T_mid)
        except Exception as e:
            print(f"  ERROR for {name}: {e}")

    # Check isotope fractionation ordering at HELIX operating T
    print("\n\nIsotope fractionation at T = 623 K (Pd-25Ag):")
    mat = get_material('Pd25Ag')
    phis = permeability_all_species(mat, 623.0)
    for sp, phi in phis.items():
        print(f"  Phi_{sp} = {phi:.3e}  mol/m/s/Pa⁰·⁵")
    print(f"  Phi_H2 / Phi_T2 = {phis['H2']/phis['T2']:.4f}")

    # Verify Phi = D * KS
    chk = check_phi_consistency(get_material('Pd25Ag'), 623.0)
    print(f"\nPhi vs D*KS consistency (Pd-25Ag, 623 K):")
    print(f"  Phi stored  = {chk['Phi_stored']:.4e}")
    print(f"  D * KS      = {chk['Phi_from_DKS']:.4e}")
    print(f"  Discrepancy = {chk['relative_discrepancy']:.1%}")
