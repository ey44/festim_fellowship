"""
material_library.py
===================
Material properties for FESTIM, transcribed from:

    M. Shimada (2020), "Tritium Transport in Fusion Reactor Materials",
    Comprehensive Nuclear Materials 2nd ed., Chapter 6.08,
    Table 1 (pure metals & carbon, p. 258) and Table 2 (alloys, p. 266).

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    import festim as F
    from material_library import Zr, Li2O

    # attributes are already in FESTIM units (m^2/s, eV, atoms/m^3/Pa^0.5)
    coating = F.Material(D_0=Zr.D_0, E_D=Zr.E_D,
                         K_S_0=Zr.K_S_0, E_K_S=Zr.E_K_S,
                         solubility_law="sievert")

    # or, equivalently, in one shot:
    coating = F.Material(**Zr.festim_kwargs(), solubility_law="sievert")

Every material defaults to the TRITIUM isotope (the 1/sqrt(m_Q) factor is
already folded into D_0). For another isotope:

    Zr_D = Zr.as_isotope("D")      # -> D_0 scaled by 1/sqrt(2) instead
    Zr.D_0_for("H")                # -> just the number

For a FESTIM multi-species problem, the dict-valued form:

    F.Material(**Zr.festim_kwargs_multispecies(("D", "T")))

--------------------------------------------------------------------------
STORAGE UNITS  (exactly as printed in the paper -- do NOT edit these)
--------------------------------------------------------------------------
    D0    [m^2/s]                    pre-exponential, diffusivity
    ED    [kJ/mol]                   activation energy, diffusivity
    KS0   [mol(Q2)/m^3/Pa^0.5]       pre-exponential, Sieverts solubility
    ES    [kJ/mol]                   enthalpy of solution
    P0    [mol(Q2)/m/s/Pa^0.5]       pre-exponential, permeability
    EP    [kJ/mol]                   activation energy, permeability

Paper's functional forms (note the 1/sqrt(m_Q) isotope factor on D and P,
and its ABSENCE on K_S):

    D_Q(T) = (1/sqrt(m_Q)) * D0  * exp(-ED / (R*T))
    K_S(T) =                 KS0 * exp(-ES / (R*T))      (isotope independent)
    P_Q(T) = (1/sqrt(m_Q)) * P0  * exp(-EP / (R*T))

with m_Q the hydrogen-isotope ATOM mass in amu (H=1, D=2, T=3), and by
construction P0 = D0 * KS0 and EP = ED + ES.

--------------------------------------------------------------------------
FESTIM UNITS -- the conversion, stated explicitly
--------------------------------------------------------------------------
FESTIM's Material (festim/material.py) uses:

    D   = D_0   * exp(-E_D   / (k_B * T))    D_0   [m^2/s], E_D   [eV]
    K_S = K_S_0 * exp(-E_K_S / (k_B * T))    K_S_0 [atoms/m^3/Pa^0.5], E_K_S [eV]

so three conversions are applied, all via named constants below:

  1. ENERGY       kJ/mol -> eV          * KJ_PER_MOL_TO_EV  (= 1/96.485)
  2. SOLUBILITY   mol(Q2)/m^3/Pa^0.5 -> atoms/m^3/Pa^0.5
                                        * MOL_H2_TO_ATOMS   (= 2*N_A)
     factor 2 because one mole of Q2 dissolves as two Q ATOMS, and FESTIM
     concentrations are atoms/m^3
  3. ISOTOPE      D_0 -> D_0 / sqrt(A_Q), because FESTIM has no isotope
                                        concept of its own

`selftest_conversions()` (and running this file directly) verifies 1-3
numerically by evaluating both forms and comparing.

--------------------------------------------------------------------------
NOTES ON THE DATA
--------------------------------------------------------------------------
  * Temperature ranges are stored PER PROPERTY. The paper gives different
    validity ranges for D, K_S and P for most materials, e.g. Pd: D 223-873 K
    but K_S and P only 473-673 K. Accessing a property outside its range
    warns.
  * Isotope scaling uses mass NUMBERS 1/2/3, matching the paper's stated
    D_H : D_D : D_T = 1 : 1/sqrt(2) : 1/sqrt(3).
  * Entries carrying `placeholder=True` fields are NOT from the paper and
    are flagged at import. Fix them before trusting any result.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────

R_GAS = 8.314462618e-3      # kJ/mol/K   gas constant, paper's Arrhenius form
N_A = 6.02214076e23         # 1/mol      Avogadro
K_B_EV = 8.617333262e-5     # eV/K       Boltzmann, FESTIM's Arrhenius form

# ── EXPLICIT UNIT CONVERSIONS (paper -> FESTIM) ───────────────────────────────
# These two constants are the entire paper->FESTIM unit bridge.

KJ_PER_MOL_TO_EV = 1.0 / 96.48533212  # [eV] per [kJ/mol] = 1000 / (N_A * e)
MOL_H2_TO_ATOMS = 2.0 * N_A           # [atoms] per [mol of Q2] = 1.20443e24

# Isotope mass numbers for the 1/sqrt(m_Q) scaling (paper, Table 1 note a)
ISOTOPE_MASS_NUMBER = {"H": 1.0, "D": 2.0, "T": 3.0}

DEFAULT_ISOTOPE = "T"


class IncompleteMaterialWarning(UserWarning):
    """Entry is missing solubility, or carries placeholder (non-paper) data."""


class PropertyRangeWarning(UserWarning):
    """Property evaluated outside its experimentally validated range."""


def _mass_factor(isotope: str) -> float:
    """1/sqrt(m_Q) with m_Q the isotope mass number (H=1, D=2, T=3)."""
    if isotope not in ISOTOPE_MASS_NUMBER:
        raise ValueError(f"isotope must be 'H', 'D' or 'T', got {isotope!r}")
    return 1.0 / np.sqrt(ISOTOPE_MASS_NUMBER[isotope])


# ── The material object ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class MaterialProperties:
    """
    One material, stored in PAPER units, exposed in FESTIM units.

    Paper-unit fields (raw, as tabulated)
        D0, ED, T_D          diffusivity     [m^2/s], [kJ/mol], (Tmin, Tmax) K
        KS0, ES, T_KS        solubility      [mol(Q2)/m^3/Pa^0.5], [kJ/mol]
        P0, EP, T_P          permeability    [mol(Q2)/m/s/Pa^0.5], [kJ/mol]

    FESTIM-unit attributes (converted on access, ready for F.Material)
        D_0    [m^2/s]                 = D0 / sqrt(A_Q)
        E_D    [eV]                    = ED * KJ_PER_MOL_TO_EV
        K_S_0  [atoms/m^3/Pa^0.5]      = KS0 * MOL_H2_TO_ATOMS
        E_K_S  [eV]                    = ES * KJ_PER_MOL_TO_EV
    """

    name: str
    D0: float
    ED: float
    T_D: tuple | None = None
    KS0: float | None = None
    ES: float | None = None
    T_KS: tuple | None = None
    P0: float | None = None
    EP: float | None = None
    T_P: tuple | None = None
    isotope: str = DEFAULT_ISOTOPE
    source: str = ""
    note: str = ""
    # names of fields that are guesses rather than tabulated data
    placeholder: tuple = field(default=())

    # ---- FESTIM-unit attributes ---------------------------------------------

    @property
    def D_0(self) -> float:
        """Pre-exponential diffusivity for self.isotope [m^2/s]. FESTIM D_0."""
        return self.D0 * _mass_factor(self.isotope)

    @property
    def E_D(self) -> float:
        """Diffusivity activation energy [eV]. FESTIM E_D."""
        return self.ED * KJ_PER_MOL_TO_EV

    @property
    def K_S_0(self) -> float:
        """Pre-exponential solubility [atoms/m^3/Pa^0.5]. FESTIM K_S_0."""
        self._require_solubility()
        return self.KS0 * MOL_H2_TO_ATOMS

    @property
    def E_K_S(self) -> float:
        """Enthalpy of solution [eV]. FESTIM E_K_S."""
        self._require_solubility()
        return self.ES * KJ_PER_MOL_TO_EV

    # ---- isotope handling ----------------------------------------------------

    def as_isotope(self, isotope: str) -> "MaterialProperties":
        """Return a copy with the diffusivity scaled for another isotope."""
        _mass_factor(isotope)  # validates
        return replace(self, isotope=isotope)

    def D_0_for(self, isotope: str) -> float:
        """FESTIM D_0 for a given isotope [m^2/s], without copying the object."""
        return self.D0 * _mass_factor(isotope)

    # ---- FESTIM hand-off -----------------------------------------------------

    def festim_kwargs(self, include_name: bool = True) -> dict:
        """
        kwargs for `festim.Material`, in FESTIM units.
        Omits K_S_0/E_K_S entirely if the material has no solubility data
        (rather than silently substituting zero).
        """
        kw = {"D_0": self.D_0, "E_D": self.E_D}
        if self.has_solubility:
            kw["K_S_0"] = self.K_S_0
            kw["E_K_S"] = self.E_K_S
        if include_name:
            kw["name"] = self.name
        return kw

    def festim_kwargs_multispecies(self, isotopes=("D", "T"),
                                   include_name: bool = True) -> dict:
        """Dict-valued kwargs for a FESTIM multi-species problem."""
        kw = {
            "D_0": {q: self.D_0_for(q) for q in isotopes},
            "E_D": {q: self.E_D for q in isotopes},
        }
        if self.has_solubility:
            kw["K_S_0"] = {q: self.K_S_0 for q in isotopes}
            kw["E_K_S"] = {q: self.E_K_S for q in isotopes}
        if include_name:
            kw["name"] = self.name
        return kw

    def to_festim(self, solubility_law: str = "sievert", **kwargs):
        """
        Build a `festim.Material` directly. FESTIM is imported lazily so this
        module stays usable (unit checks, plots) without a dolfinx install.
        Extra kwargs (thermal_conductivity, density, ...) pass through in SI.
        """
        kw = self.festim_kwargs()
        kw.update(kwargs)
        if "K_S_0" not in kw and solubility_law != "none":
            raise ValueError(
                f"{self.name} has no solubility data, so solubility_law must "
                f"be 'none' (single-material problems only), or you must pass "
                f"K_S_0 and E_K_S explicitly in FESTIM units "
                f"(atoms/m^3/Pa^0.5 and eV). See "
                f"{self.name}.with_solubility(...) to set a placeholder."
            )

        import festim as F

        return F.Material(solubility_law=solubility_law, **kw)

    # ---- editing / placeholders ---------------------------------------------

    def with_solubility(self, KS0: float, ES: float, T_KS: tuple | None = None,
                        placeholder: bool = True) -> "MaterialProperties":
        """
        Return a copy with solubility filled in, in PAPER units
        (KS0 [mol(Q2)/m^3/Pa^0.5], ES [kJ/mol]). Permeability is derived as
        P0 = D0*KS0, EP = ED+ES, per the paper's own identity.

        `placeholder=True` (default) marks the values as a guess so they show
        up in the import-time warning. Set False only for sourced data.
        """
        marks = set(self.placeholder)
        if placeholder:
            marks.update({"KS0", "ES"})
        else:
            marks -= {"KS0", "ES"}
        return replace(
            self,
            KS0=KS0, ES=ES, T_KS=T_KS,
            P0=self.D0 * KS0, EP=self.ED + ES, T_P=T_KS,
            placeholder=tuple(sorted(marks)),
        )

    # ---- properties in PAPER units ------------------------------------------

    @property
    def has_solubility(self) -> bool:
        return self.KS0 is not None

    def _require_solubility(self) -> None:
        if not self.has_solubility:
            raise ValueError(
                f"{self.name} has no solubility data (diffusivity-only entry). "
                f"Use {self.name}.with_solubility(KS0=..., ES=...) to set a "
                f"placeholder in paper units, or supply K_S_0/E_K_S directly "
                f"in FESTIM units."
            )

    def _check_range(self, T: float, key: str, label: str) -> None:
        rng = getattr(self, key)
        if rng is None:
            return
        lo, hi = rng
        if not lo <= T <= hi:
            warnings.warn(
                f"{self.name}: T = {T} K is outside the validated range "
                f"{lo}-{hi} K for {label}; Arrhenius extrapolation may be "
                f"unreliable.",
                PropertyRangeWarning, stacklevel=3,
            )

    def diffusivity(self, T: float, isotope: str | None = None) -> float:
        """D_Q(T) = (1/sqrt(m_Q)) * D0 * exp(-ED/(R*T))   [m^2/s]"""
        self._check_range(T, "T_D", "diffusivity")
        q = self.isotope if isotope is None else isotope
        return self.D0 * _mass_factor(q) * np.exp(-self.ED / (R_GAS * T))

    def sieverts_constant(self, T: float) -> float:
        """K_S(T) = KS0*exp(-ES/(R*T)) [mol(Q2)/m^3/Pa^0.5], isotope independent."""
        self._require_solubility()
        self._check_range(T, "T_KS", "solubility")
        return self.KS0 * np.exp(-self.ES / (R_GAS * T))

    def permeability(self, T: float, isotope: str | None = None) -> float:
        """P_Q(T) = (1/sqrt(m_Q)) * P0 * exp(-EP/(R*T)) [mol(Q2)/m/s/Pa^0.5]"""
        if self.P0 is None:
            raise ValueError(f"{self.name} has no permeability data.")
        self._check_range(T, "T_P", "permeability")
        q = self.isotope if isotope is None else isotope
        return self.P0 * _mass_factor(q) * np.exp(-self.EP / (R_GAS * T))

    # ---- checks --------------------------------------------------------------

    def check_phi_consistency(self, T: float) -> dict:
        """
        Verify the paper's own identity P_Q = D_Q * K_S at T. A nonzero
        discrepancy means the tabulated P0/EP were not derived from the same
        D0/KS0 pair (mixed source datasets, or table rounding).
        """
        self._require_solubility()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PropertyRangeWarning)
            P = self.permeability(T)
            P_dks = self.diffusivity(T) * self.sieverts_constant(T)
        rel = abs(P - P_dks) / max(P, P_dks)
        return {"T": T, "P_stored": P, "P_from_DKS": P_dks,
                "relative_discrepancy": rel, "consistent": rel < 0.01}

    def check_festim_roundtrip(self, T: float) -> dict:
        """
        Evaluate D and K_S both in paper units and via the FESTIM (eV/atoms)
        parameters, and confirm they agree. Catches unit-conversion slips.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PropertyRangeWarning)
            D_paper = self.diffusivity(T)
            D_festim = self.D_0 * np.exp(-self.E_D / (K_B_EV * T))
            out = {"D_rel_err": abs(D_festim - D_paper) / D_paper,
                   "KS_rel_err": None}
            if self.has_solubility:
                KS_paper = self.sieverts_constant(T)
                KS_festim = self.K_S_0 * np.exp(-self.E_K_S / (K_B_EV * T))
                out["KS_rel_err"] = (
                    abs(KS_festim / MOL_H2_TO_ATOMS - KS_paper) / KS_paper
                )
        return out

    def summary(self, T: float) -> str:
        def rng(key):
            r = getattr(self, key)
            return f"valid {r[0]}-{r[1]} K" if r else "range unspecified"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PropertyRangeWarning)
            lines = [
                f"{'=' * 70}",
                f"{self.name}   isotope {self.isotope}   "
                f"T = {T:.0f} K ({T - 273.15:.0f} degC)",
                f"{'=' * 70}",
                "  -- paper units --",
                f"    D    = {self.diffusivity(T):.3e}  m^2/s              "
                f"({rng('T_D')})",
            ]
            if self.has_solubility:
                chk = self.check_phi_consistency(T)
                lines += [
                    f"    K_S  = {self.sieverts_constant(T):.3e}  "
                    f"mol(Q2)/m^3/Pa^0.5 ({rng('T_KS')})",
                    f"    P    = {self.permeability(T):.3e}  "
                    f"mol(Q2)/m/s/Pa^0.5 ({rng('T_P')})",
                    f"    P vs D*K_S: {chk['relative_discrepancy']:.2%}"
                    + ("  ok" if chk["consistent"] else "  <-- MIXED DATASETS"),
                ]
            else:
                lines.append("    K_S / P = (none)  diffusivity-only entry")
            lines += [
                "  -- FESTIM units --",
                f"    D_0   = {self.D_0:.4e}  m^2/s",
                f"    E_D   = {self.E_D:.4f}  eV",
            ]
            if self.has_solubility:
                lines += [f"    K_S_0 = {self.K_S_0:.4e}  atoms/m^3/Pa^0.5",
                          f"    E_K_S = {self.E_K_S:.4f}  eV"]
            else:
                lines.append("    K_S_0 = (omitted; supply before coupling)")
            if self.placeholder:
                lines.append(
                    f"  !! PLACEHOLDER FIELDS: {', '.join(self.placeholder)}"
                )
            if self.source:
                lines.append(f"  source: {self.source}")
            if self.note:
                lines.append(f"  note: {self.note}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        tag = "" if self.has_solubility else ", D-only"
        ph = f", placeholder={list(self.placeholder)}" if self.placeholder else ""
        return (f"MaterialProperties({self.name!r}, isotope={self.isotope!r}, "
                f"D_0={self.D_0:.3e} m^2/s, E_D={self.E_D:.4f} eV{tag}{ph})")


# ── The materials ─────────────────────────────────────────────────────────────
# Import these directly:  from material_library import Zr, Li2O, Pd25Ag

Pd25Ag = MaterialProperties(
    name="Pd25Ag",
    D0=3.07e-7, ED=25.9, T_D=(323, 773),
    KS0=1.82e-1, ES=-19.6, T_KS=(323, 773),
    P0=5.58e-8, EP=6.3, T_P=(323, 773),
    source="Shimada Table 2; Serra et al. (1998) Metall. Mater. Trans. A 29A, 1023",
    note=("Pd/Ag membrane alloy. ES negative: solubility rises on cooling. "
          "Self-consistent set (P0 = D0*KS0, EP = ED + ES)."),
)

Pd = MaterialProperties(
    name="Pd",
    D0=2.90e-7, ED=22.2, T_D=(223, 873),
    KS0=4.45e-1, ES=-8.4, T_KS=(473, 673),
    P0=1.29e-7, EP=13.8, T_P=(473, 673),
    source=("Shimada Table 1; D: Volkl & Alefeld (1975); "
            "K_S: Favreau et al. (1954)"),
    note=("Pure Pd. D valid over a far wider range (223-873 K) than K_S and P "
          "(473-673 K) -- do not extrapolate solubility."),
)

Nb = MaterialProperties(
    name="Nb",
    D0=5.00e-8, ED=10.2, T_D=(223, 873),
    KS0=1.26e-1, ES=-35.3, T_KS=(625, 944),
    P0=6.30e-9, EP=-25.1, T_P=(625, 773),
    source=("Shimada Table 1; D: Volkl & Alefeld (1975); "
            "K_S: Veleckis & Edwards (1969)"),
    note=("Group V metal, very high bulk diffusivity but strongly surface "
          "sensitive -- oxygen contamination makes permeation data unreliable "
          "(Shimada Sec. 6.08.3.12). EP negative -> P rises on cooling."),
)

V = MaterialProperties(
    name="V",
    D0=2.90e-8, ED=4.2, T_D=(173, 573),
    KS0=1.38e-1, ES=-29.0, T_KS=(519, 827),
    P0=4.00e-9, EP=-24.9, T_P=(519, 573),
    source=("Shimada Table 1; D: Volkl & Alefeld (1975); "
            "K_S: Veleckis & Edwards (1969)"),
    note=("Very high diffusivity, highly surface sensitive. D and K_S ranges "
          "barely overlap; P validated only over 519-573 K. The ~2% P vs "
          "D*K_S mismatch is table rounding (EP -24.9 vs ED+ES -24.8)."),
)

Zr = MaterialProperties(
    name="Zr",
    D0=8.00e-7, ED=45.3, T_D=(548, 973),
    KS0=4.30e-1, ES=-49.5, T_KS=(602, 1069),
    P0=3.44e-7, EP=-4.2, T_P=(602, 973),
    source="Shimada Table 1; D: Kearns (1967); K_S: Kearns (1972)",
    note=("Pebble coating candidate. Strongly negative ES -- Zr is a hydride "
          "former and dissolves a lot of hydrogen; this Sieverts fit is a "
          "dilute-solution approximation that breaks down at high loading."),
)

Li2O = MaterialProperties(
    name="Li2O",
    D0=1.16e-5, ED=101.0, T_D=None,
    source=("NOT from Shimada (2020) -- Ch. 6.08 covers metals, alloys, carbon "
            "and Pb-17Li only, with no ceramic breeder data. User-supplied; "
            "add the primary reference."),
    note=("Breeder pebble core. Diffusivity only. FESTIM multi-material "
          "problems need a solubility law on EVERY material for the interface "
          "condition, so set K_S before coupling to the coating: "
          "Li2O = Li2O.with_solubility(KS0=..., ES=...)"),
)

RAFM = MaterialProperties(
    name="RAFM",
    D0=1.00e-7, ED=13.2, T_D=(300, 973),
    KS0=4.40e-1, ES=28.6, T_KS=(300, 973),
    P0=4.40e-8, EP=41.8, T_P=(300, 973),
    source="Shimada Table 2; Causey, Karnesky & San Marchi (2012) CNM Ch. 4.16",
    note=("F82H / Eurofer. Below ~573 K trapping makes the EFFECTIVE D lower "
          "and effective K_S higher than these lattice values -- use above "
          "573 K, or model traps explicitly."),
)

SS316L = MaterialProperties(
    name="SS316L",
    D0=8.70e-7, ED=51.9, T_D=(500, 1200),
    KS0=3.60e-1, ES=11.7, T_KS=(500, 1200),
    P0=3.13e-7, EP=63.6, T_P=(500, 1200),
    source="Shimada Table 2; Reiter, Forcey & Gervasini (1993) EUR 15217 EN",
    note="Loop tubing. High EP -> good tritium barrier at ambient T.",
)


MATERIALS = {m.name: m for m in
             (Pd25Ag, Pd, Nb, V, Zr, Li2O, RAFM, SS316L)}


def get_material(name: str) -> MaterialProperties:
    """Look a material up by name."""
    if name not in MATERIALS:
        raise ValueError(
            f"Material {name!r} not in library. Available: {list(MATERIALS)}"
        )
    return MATERIALS[name]


# ── Import-time flag for incomplete / placeholder entries ─────────────────────

def incomplete_materials() -> dict:
    """{name: [problem, ...]} for every entry that is not fully sourced."""
    out = {}
    for name, m in MATERIALS.items():
        problems = []
        if not m.has_solubility:
            problems.append("no K_S or permeability")
        if m.placeholder:
            problems.append(f"placeholder {', '.join(m.placeholder)}")
        if problems:
            out[name] = problems
    return out


def _warn_incomplete() -> None:
    gaps = incomplete_materials()
    if not gaps:
        return
    warnings.warn(
        "material_library: "
        + "; ".join(f"{n} ({', '.join(p)})" for n, p in gaps.items())
        + ". Materials without K_S expose D_0/E_D only -- K_S_0/E_K_S raise, "
          "and festim_kwargs() omits them. FESTIM multi-material problems "
          "require a solubility law on every material.",
        IncompleteMaterialWarning, stacklevel=2,
    )


_warn_incomplete()


# ── Self-test ─────────────────────────────────────────────────────────────────

def selftest_conversions(T: float = 573.0) -> None:
    print(f"\n{'=' * 70}\nUnit-conversion round trip at {T:.0f} K "
          f"(errors should be ~0)\n{'=' * 70}")
    for name, m in MATERIALS.items():
        r = m.check_festim_roundtrip(T)
        ks = "n/a" if r["KS_rel_err"] is None else f"{r['KS_rel_err']:.2e}"
        print(f"  {name:<8} D err = {r['D_rel_err']:.2e}   K_S err = {ks}")


if __name__ == "__main__":
    print("material_library self-test -- Shimada (2020) Tables 1 & 2")
    print(f"KJ_PER_MOL_TO_EV = {KJ_PER_MOL_TO_EV:.6e} eV per kJ/mol")
    print(f"MOL_H2_TO_ATOMS  = {MOL_H2_TO_ATOMS:.6e} atoms per mol(Q2)")
    print(f"default isotope  = {DEFAULT_ISOTOPE}")

    for m in MATERIALS.values():
        ranges = [r for r in (m.T_D, m.T_KS, m.T_P) if r]
        T_eval = (0.5 * (max(r[0] for r in ranges) + min(r[1] for r in ranges))
                  if ranges else 573.0)
        print("\n" + m.summary(T_eval))

    selftest_conversions()

    print(f"\n{'=' * 70}\nIsotope scaling, Pd-25Ag at 623 K "
          f"(expect 1 : 0.707 : 0.577)\n{'=' * 70}")
    d_h = Pd25Ag.diffusivity(623.0, "H")
    for q in ("H", "D", "T"):
        d = Pd25Ag.diffusivity(623.0, q)
        print(f"  D_{q} = {d:.4e} m^2/s   ratio {d / d_h:.4f}")

    print(f"\n{'=' * 70}\nPlaceholder workflow for Li2O\n{'=' * 70}")
    print(f"  before: {Li2O!r}")
    demo = Li2O.with_solubility(KS0=1.0e-1, ES=0.0)
    print(f"  after : {demo!r}")
    print(f"  K_S_0 = {demo.K_S_0:.4e} atoms/m^3/Pa^0.5   "
          f"E_K_S = {demo.E_K_S:.4f} eV")
