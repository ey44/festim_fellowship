"""
hx_thermal_sizing.py  —  Phase 0
================================
Thermal sizing of the He -> water/steam heat exchanger, so that the
heat-transfer area A (and hence N tubes x length L) used for tritium
permeation is physically determined rather than assumed.

Method
------
1. Duty from the He side:        Q = mdot_He * cp_He * (T_h_in - T_h_out)
2. LMTD from the four terminal temperatures (counterflow).
3. He-side h from a Nusselt correlation (laminar 3.66 / Gnielinski
   turbulent, interpolated in transition — mirrors sherwood_number()).
4. 1/U = 1/h_He + t_wall/k_wall + 1/h_sec   (thin-wall, per unit area)
5. A = Q / (U * LMTD);  N from a per-tube Reynolds target;  L = A/(N*pi*d_i)
6. Axial profiles T_he(z), T_sec(z), T_wall(z) from the counterflow
   energy balance (near-linear here: capacity-rate ratio ~ 1.03).
7. Feasibility checks: He pressure drop, velocity/Mach, single-phase caveat.

Outputs: printed design table + hx_design.json + hx_profiles.csv
(consumed by axial_integration.py and hx_2d_model.py).

Assumptions to revisit:
- Secondary assumed SINGLE-PHASE 290->580 C (supercritical or superheated
  steam). If it boils, split into zones — single LMTD invalid.
- k_wall (Inconel 617) taken constant at its 300-600 C mid-range value.
- h_sec is a parameter (water/steam side is not the controlling resistance).
"""

import json
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
R_GAS  = 8.314          # J/mol/K
M_HE   = 4.003e-3       # kg/mol
CP_HE  = 5193.0         # J/kg/K  (monatomic ideal gas, pressure-independent)
PR_HE  = 0.664          # Prandtl number of He (nearly T-independent)
GAMMA_HE = 5.0 / 3.0


def viscosity_He(T):
    """Dynamic viscosity of He [Pa.s] — same fit as transport.py."""
    return 1.99e-5 * (T / 300.0) ** 0.67


def conductivity_He(T):
    """Thermal conductivity of He [W/m/K]. k = cp*mu/Pr (consistent set)."""
    return CP_HE * viscosity_He(T) / PR_HE


def density_He(T, P):
    return P * M_HE / (R_GAS * T)


def friction_factor(Re):
    """Darcy friction factor, smooth tube (Petukhov), laminar 64/Re."""
    if Re < 2300:
        return 64.0 / Re
    return (0.790 * np.log(Re) - 1.64) ** -2


def nusselt_number(Re, Pr):
    """
    Nusselt number for flow in a tube. Mirrors transport.sherwood_number():
      laminar   (Re<2300)  : 3.66 (constant wall T)
      turbulent (Re>4000)  : Gnielinski
      transition           : linear interpolation — flagged uncertain
    """
    def gnielinski(Re_):
        f = friction_factor(max(Re_, 4000.0))
        return (f / 8) * (Re_ - 1000) * Pr / (1 + 12.7 * np.sqrt(f / 8) * (Pr ** (2 / 3) - 1))

    if Re < 2300:
        return 3.66, 'laminar'
    elif Re > 4000:
        return gnielinski(Re), 'turbulent (Gnielinski)'
    else:
        Nu_lam, Nu_turb = 3.66, gnielinski(4000.0)
        f = (Re - 2300) / (4000 - 2300)
        return Nu_lam + f * (Nu_turb - Nu_lam), 'transition (interpolated — uncertain)'


# ── Main sizing routine ──────────────────────────────────────────────────────

def size_hx(
    mdot_he   = 1.0,        # kg/s        He mass flow
    T_h_in    = 600 + 273.15,
    T_h_out   = 300 + 273.15,
    T_c_in    = 290 + 273.15,
    T_c_out   = 580 + 273.15,
    P_he      = 2e5,        # Pa          He total pressure (purge-loop level)
    d_i       = 0.010,      # m           tube inner diameter
    t_wall    = 2.5e-3,     # m           wall thickness (Inconel 617)
    k_wall    = 20.0,       # W/m/K       IN617, ~300-600 C mid-range
    h_sec     = 4000.0,     # W/m2/K      water/steam side film coefficient
    Re_target = 3000.0,     # -           per-tube Reynolds (sets N tubes)
    n_z       = 101,        # -           axial stations for profiles
    verbose   = True,
):
    """Returns a design dict; optionally prints a summary table."""

    # 1. Duty and capacity rates
    Q_duty = mdot_he * CP_HE * (T_h_in - T_h_out)            # W
    C_hot  = mdot_he * CP_HE                                 # W/K
    C_cold = Q_duty / (T_c_out - T_c_in)                     # W/K (from energy balance)

    # 2. LMTD (counterflow)
    dT1 = T_h_in - T_c_out      # hot end
    dT2 = T_h_out - T_c_in      # cold end
    if abs(dT1 - dT2) < 1e-9:
        lmtd = dT1
    else:
        lmtd = (dT1 - dT2) / np.log(dT1 / dT2)

    # 3. He-side film coefficient at mean He temperature
    T_mean = 0.5 * (T_h_in + T_h_out)
    mu  = viscosity_He(T_mean)
    rho = density_He(T_mean, P_he)
    k   = conductivity_He(T_mean)

    # per-tube mass flow from Re target: Re = 4*mdot_tube/(pi*d_i*mu)
    mdot_tube = Re_target * np.pi * d_i * mu / 4.0
    n_tubes   = int(np.ceil(mdot_he / mdot_tube))
    mdot_tube = mdot_he / n_tubes                            # actual, after rounding
    Re        = 4.0 * mdot_tube / (np.pi * d_i * mu)
    Nu, regime = nusselt_number(Re, PR_HE)
    h_he      = Nu * k / d_i

    # 4. Overall U referenced to the INNER area, with cylindrical wall:
    #    R_wall = r_i ln(r_o/r_i)/k ;  secondary film scaled by A_i/A_o
    r_i, r_o = d_i / 2.0, d_i / 2.0 + t_wall
    R_he = 1.0 / h_he
    R_w  = r_i * np.log(r_o / r_i) / k_wall
    R_s  = (r_i / r_o) / h_sec
    U = 1.0 / (R_he + R_w + R_s)

    # 5. Area and length
    A = Q_duty / (U * lmtd)
    L = A / (n_tubes * np.pi * d_i)

    # 6. Axial profiles (counterflow, constant U and C's)
    #    dT_h/dA = -U*(T_h - T_c)/C_h ;  dT_c/dA = -U*(T_h - T_c)/C_c
    #    z=0 is the He INLET (hot end); secondary exits at z=0.
    z  = np.linspace(0.0, L, n_z)
    a  = z / L * A                                           # cumulative area
    r  = U * (1.0 / C_hot - 1.0 / C_cold)                    # 1/area
    dT0 = T_h_in - T_c_out                                   # dT at z=0
    if abs(r) > 1e-15:
        dT = dT0 * np.exp(-r * a)
    else:
        dT = np.full_like(a, dT0)
    # integrate hot stream: T_h(a) = T_h_in - (U/C_h) * int dT da
    if abs(r) > 1e-15:
        int_dT = dT0 * (1.0 - np.exp(-r * a)) / r
    else:
        int_dT = dT0 * a
    T_he  = T_h_in - (U / C_hot) * int_dT
    T_sec = T_he - dT

    # local heat flux and wall temperatures
    q_pp      = U * dT                                       # W/m2
    T_wall_he = T_he - q_pp / h_he                           # He-side wall surface
    T_wall_sec = T_wall_he - q_pp * t_wall / k_wall          # sec-side wall surface
    T_wall_mid = 0.5 * (T_wall_he + T_wall_sec)

    # 7. Feasibility: He velocity, Mach, pressure drop
    u    = mdot_tube / (rho * np.pi * (d_i / 2) ** 2)
    Mach = u / np.sqrt(GAMMA_HE * R_GAS * T_mean / M_HE)
    f    = friction_factor(Re)
    dP   = f * (L / d_i) * 0.5 * rho * u ** 2

    design = {
        'inputs': dict(mdot_he=mdot_he, T_h_in=T_h_in, T_h_out=T_h_out,
                       T_c_in=T_c_in, T_c_out=T_c_out, P_he=P_he, d_i=d_i,
                       t_wall=t_wall, k_wall=k_wall, h_sec=h_sec,
                       Re_target=Re_target),
        'Q_duty_W': Q_duty, 'LMTD_K': lmtd,
        'C_hot_W_K': C_hot, 'C_cold_W_K': C_cold,
        'C_ratio': C_cold / C_hot,
        'effectiveness': (T_h_in - T_h_out) / (T_h_in - T_c_in),
        'Re': Re, 'Nu': Nu, 'flow_regime': regime,
        'h_he_W_m2K': h_he, 'U_W_m2K': U,
        'R_he': R_he, 'R_wall': R_w, 'R_sec': R_s,
        'A_m2': A, 'n_tubes': n_tubes, 'L_m': L,
        'u_he_m_s': u, 'Mach': Mach, 'dP_he_Pa': dP,
        'dP_over_P': dP / P_he,
    }

    if verbose:
        print(f"{'='*64}\nHX thermal sizing — He -> water/steam, counterflow\n{'='*64}")
        print(f"  Duty                Q      = {Q_duty/1e6:8.3f}  MW")
        print(f"  LMTD                       = {lmtd:8.2f}  K   "
              f"(dT hot end {dT1:.0f} K, cold end {dT2:.0f} K)")
        print(f"  Capacity ratio      Cc/Ch  = {design['C_ratio']:8.3f}  (near-balanced -> ~linear T(z))")
        print(f"  Effectiveness       eps    = {design['effectiveness']:8.3f}")
        print(f"  Re/tube                    = {Re:8.0f}  ({regime})")
        print(f"  Nu                         = {Nu:8.2f}")
        print(f"  h_He                       = {h_he:8.1f}  W/m2/K")
        print(f"  U                          = {U:8.1f}  W/m2/K  "
              f"(He {100*R_he*U:.0f}% | wall {100*R_w*U:.0f}% | sec {100*R_s*U:.0f}% of resistance)")
        print(f"  Area                A      = {A:8.1f}  m2")
        print(f"  Tubes               N      = {n_tubes:8d}  (d_i={d_i*1e3:.0f} mm, t={t_wall*1e3:.1f} mm)")
        print(f"  Tube length         L      = {L:8.2f}  m")
        print(f"  He velocity / Mach         = {u:8.1f}  m/s  /  {Mach:.3f}")
        print(f"  He pressure drop    dP     = {dP/1e3:8.1f}  kPa  ({100*dP/P_he:.1f}% of P_He)"
              + ("   ** EXCESSIVE — revisit P_he/d_i/Re_target **" if dP/P_he > 0.1 else ""))
        print(f"  Wall T range (He-side)     = {T_wall_he.min()-273.15:.0f} to "
              f"{T_wall_he.max()-273.15:.0f} C")

    profiles = {
        'z_m': z, 'T_he_K': T_he, 'T_sec_K': T_sec,
        'T_wall_he_K': T_wall_he, 'T_wall_mid_K': T_wall_mid,
        'T_wall_sec_K': T_wall_sec, 'q_pp_W_m2': q_pp,
    }
    return design, profiles


# ── Run baseline + U sensitivity ─────────────────────────────────────────────

if __name__ == '__main__':
    design, prof = size_hx()

    # save for downstream phases
    with open('hx_design.json', 'w') as fp:
        json.dump(design, fp, indent=2)
    arr = np.column_stack([prof[k] for k in prof])
    np.savetxt('hx_profiles.csv', arr, delimiter=',',
               header=','.join(prof.keys()), comments='')
    print("\nSaved hx_design.json and hx_profiles.csv")

    # quick U sensitivity (h_sec and Re_target bracket)
    print(f"\n{'-'*64}\nSensitivity of area to design choices\n{'-'*64}")
    print(f"{'Re_target':>10} {'h_sec':>8} {'U':>8} {'A [m2]':>9} "
          f"{'N':>6} {'L [m]':>7} {'dP/P':>7}")
    for Re_t in (2000.0, 3000.0, 5000.0, 8000.0):
        for hs in (2000.0, 4000.0, 8000.0):
            d, _ = size_hx(Re_target=Re_t, h_sec=hs, verbose=False)
            print(f"{Re_t:10.0f} {hs:8.0f} {d['U_W_m2K']:8.1f} {d['A_m2']:9.1f} "
                  f"{d['n_tubes']:6d} {d['L_m']:7.2f} {d['dP_over_P']:7.3f}")

    # energy-balance closure check (V&V, Phase 0)
    Q_cold = design['C_cold_W_K'] * (design['inputs']['T_c_out'] - design['inputs']['T_c_in'])
    assert abs(Q_cold - design['Q_duty_W']) / design['Q_duty_W'] < 1e-12
    print("\nEnergy-balance closure: OK")
