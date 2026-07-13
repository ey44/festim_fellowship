"""
area_sizing.py
==============
Find the membrane area required to achieve a target total-stream recovery
using the permeator_v3 model.

"Total stream" recovery:
    eta_total = (F_H2_permeated + F_T2_permeated) / (F_H2_in + F_T2_in)
              = (eta_H2 * F_H2_in + eta_T2 * F_T2_in) / (F_H2_in + F_T2_in)

For x_H = 0.5 this simplifies to (eta_H2 + eta_T2) / 2.

Strategy: fix all operating parameters, vary tube length L, find the L that
hits the target using a bracketed root-find (brentq), then report required area.
"""

import numpy as np
from scipy.optimize import brentq
from permeator_v3 import solve_permeator

# ── Target and base parameters ────────────────────────────────────────────────

TARGET_ETA = 0.80      # 80% total-stream recovery

BASE_PARAMS = {}       # use permeator_v3 defaults; override here if needed
                       # e.g. {'T': 673, 'p_H_total': 2.0, 'f_phi': 0.8}


# ── Helper: total-stream recovery from a result dict ─────────────────────────

def eta_total(result):
    p     = result['params']
    F_in  = p['F_H2_in'] + p['F_T2_in']
    F_out = result['F_H2'][-1] + result['F_T2'][-1]
    return (F_in - F_out) / F_in if F_in > 0 else 0.0


# ── Root-find over L ──────────────────────────────────────────────────────────

def eta_minus_target(L, base_params, target):
    r = solve_permeator({**base_params, 'L': L})
    return eta_total(r) - target


# Bracket: find L_lo (under-recovery) and L_hi (over-recovery)
# Start with a wide bracket and home in.
print(f"Searching for L that gives η_total = {TARGET_ETA*100:.0f}%...\n")

# Quick scan to find a valid bracket
L_scan  = np.geomspace(1e-3, 1e3, 50)
eta_scan = []
for L in L_scan:
    r = solve_permeator({**BASE_PARAMS, 'L': L})
    eta_scan.append(eta_total(r))
    if eta_scan[-1] > TARGET_ETA:
        break

eta_scan = np.array(eta_scan)
L_scan   = L_scan[:len(eta_scan)]

# Check thermodynamic ceiling
r_long = solve_permeator({**BASE_PARAMS, 'L': 1e4})
eta_max = eta_total(r_long)

if eta_max < TARGET_ETA:
    print(f"  ⚠  Thermodynamic ceiling: η_max = {eta_max*100:.1f}%")
    print(f"     Cannot reach {TARGET_ETA*100:.0f}% with these operating conditions.")
    print(f"     Reduce p_perm or increase p_H_total.")
else:
    # Find bracket
    idx = np.searchsorted(eta_scan, TARGET_ETA)
    if idx == 0:
        L_lo, L_hi = L_scan[0] * 0.01, L_scan[0]
    else:
        L_lo, L_hi = L_scan[idx - 1], L_scan[idx]

    L_req = brentq(eta_minus_target, L_lo, L_hi,
                   args=(BASE_PARAMS, TARGET_ETA),
                   xtol=1e-4, rtol=1e-6)

    r_req  = solve_permeator({**BASE_PARAMS, 'L': L_req})
    p      = r_req['params']
    d      = r_req['diagnostics']
    A_req  = np.pi * p['d_i'] * L_req * p['n_tubes']   # m²
    eta_H2 = r_req['eta_H2']
    eta_T2 = r_req['eta_T2']
    eta_tot = eta_total(r_req)

    print("=" * 60)
    print(f"  Required membrane area for η_total = {TARGET_ETA*100:.0f}%")
    print("=" * 60)
    print(f"\n  Operating conditions:")
    print(f"    T          = {p['T']:.0f} K  ({p['T']-273.15:.0f} °C)")
    print(f"    P_He       = {p['P_He']/1e5:.2f} bar")
    print(f"    p_H_total  = {p['p_H_total']:.2f} Pa")
    print(f"    p_perm     = {p['p_perm']:.3f} Pa")
    print(f"    F_H_total  = {p['F_H_total_in']*1e6:.2f} µmol/s")
    print(f"    x_H_in     = {p['x_H_in']:.2f}")
    print(f"\n  Geometry at target:")
    print(f"    n_tubes    = {p['n_tubes']}")
    print(f"    d_i        = {p['d_i']*1e3:.1f} mm")
    print(f"    delta      = {p['delta']*1e6:.0f} µm")
    print(f"    L_required = {L_req:.4f} m")
    print(f"    A_required = {A_req:.4f} m²  ({A_req*1e4:.2f} cm²)")
    print(f"\n  Recovery at target area:")
    print(f"    η_H2       = {eta_H2*100:.2f}%")
    print(f"    η_T2       = {eta_T2*100:.2f}%")
    print(f"    η_total    = {eta_tot*100:.2f}%")
    print(f"\n  Regime (inlet → exit):")
    print(f"    Λ_inlet    = {d['Lambda_inlet']:.3f}")
    print(f"    Λ_exit     = {d['Lambda_exit']:.3f}")
    print(f"    Thermodynamic ceiling: η_max = {eta_max*100:.1f}%")

    # ── Standard tube count ───────────────────────────────────────────────────
    L_std  = 0.5          # m   standard tube length
    d_std  = 0.010        # m   standard tube diameter (10 mm)
    A_tube = np.pi * d_std * L_std
    n_std  = A_req / A_tube
    print(f"\n  Equivalent standard tubes (L={L_std} m, d={d_std*1e3:.0f} mm):")
    print(f"    Area per tube = {A_tube*1e4:.2f} cm²")
    print(f"    Tubes needed  = {n_std:.1f}  →  {int(np.ceil(n_std))} tubes")

    # ── Sensitivity table: area vs recovery ──────────────────────────────────
    print(f"\n  Recovery vs area (at fixed operating conditions):")
    print(f"  {'η_total [%]':>12}  {'L [m]':>10}  {'A [m²]':>10}  {'η_H2 [%]':>10}  {'η_T2 [%]':>10}")
    for target in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        if target >= eta_max:
            print(f"  {target*100:12.0f}  {'(above ceiling)':>10}")
            continue
        try:
            L_t = brentq(eta_minus_target, L_scan[0]*0.01, 1e3,
                         args=(BASE_PARAMS, target), xtol=1e-4, rtol=1e-6)
            r_t = solve_permeator({**BASE_PARAMS, 'L': L_t})
            A_t = np.pi * p['d_i'] * L_t * p['n_tubes']
            print(f"  {target*100:12.0f}  {L_t:10.4f}  {A_t:10.4f}"
                  f"  {r_t['eta_H2']*100:10.2f}  {r_t['eta_T2']*100:10.2f}")
        except Exception:
            print(f"  {target*100:12.0f}  {'(not reachable)':>10}")
