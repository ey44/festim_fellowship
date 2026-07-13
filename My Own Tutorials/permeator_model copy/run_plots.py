"""
run_plots.py
============
Generate all 16 permeator plots and save to the project directory.
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from permeator_v3 import solve_permeator
from plots import (
    plot_axial_molar_flows, plot_axial_partial_pressures,
    plot_axial_flux, plot_axial_lambda,
    plot_recovery_bars, plot_resistance_split,
    plot_temp_recovery, plot_temp_lambda,
    plot_pH_recovery, plot_pperm_recovery,
    plot_composition_map,
    plot_regime_lambda_profile, plot_regime_composition,
    plot_tornado,
    plot_geometry_pressure_profile, plot_geometry_species_profiles,
    plot_tubes_vs_pressure_map,
)

OUT = os.path.dirname(os.path.abspath(__file__)) + '/'
print(f"Saving plots to: {OUT}\n")

# ── Base result ───────────────────────────────────────────────────────────────
print("Running base case...")
r0 = solve_permeator()

# ── Axial profiles (4 figures) ────────────────────────────────────────────────
plot_axial_molar_flows(      r0, OUT + 'axial_molar_flows.png')
plot_axial_partial_pressures(r0, OUT + 'axial_partial_pressures.png')
plot_axial_flux(             r0, OUT + 'axial_flux.png')
plot_axial_lambda(           r0, OUT + 'axial_lambda.png')

# ── Recovery & resistance (2 figures) ────────────────────────────────────────
plot_recovery_bars(   r0, OUT + 'recovery_bars.png')
plot_resistance_split(r0, OUT + 'resistance_split.png')

# ── Temperature sensitivity (2 figures) ──────────────────────────────────────
print("\nRunning temperature sweep...")
T_arr = [473, 523, 573, 623, 673, 723]
r_T   = [solve_permeator({'T': T}) for T in T_arr]
plot_temp_recovery(T_arr, r_T, OUT + 'temp_recovery.png')
plot_temp_lambda(  T_arr, r_T, OUT + 'temp_lambda.png')

# ── Pressure sensitivity (2 figures) ─────────────────────────────────────────
print("\nRunning pressure sweeps...")
pH_arr = [5, 10, 25, 50, 100, 250, 500, 1000]
r_pH   = [solve_permeator({'p_H_total': p}) for p in pH_arr]
plot_pH_recovery(pH_arr, r_pH, OUT + 'pH_recovery.png')

pp_arr = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
r_pp   = [solve_permeator({'p_perm': p}) for p in pp_arr]
plot_pperm_recovery(pp_arr, r_pp, OUT + 'pperm_recovery.png')

# ── Composition map (1 figure) ────────────────────────────────────────────────
print("\nRunning composition map...")
xH_arr   = np.linspace(0.05, 0.95, 12)
T_map    = [473, 523, 573, 623, 673, 723]
eta_grid = [
    [solve_permeator({'T': T, 'x_H_in': float(xH)})['eta_T']
     for xH in xH_arr]
    for T in T_map
]
plot_composition_map(xH_arr, T_map, eta_grid, OUT + 'composition_map.png')

# ── Regime map (2 figures) ────────────────────────────────────────────────────
plot_regime_lambda_profile(r0, OUT + 'regime_lambda_profile.png')
plot_regime_composition(   r0, OUT + 'regime_composition.png')

# ── Tornado (1 figure) ────────────────────────────────────────────────────────
print("\nRunning tornado sensitivity...")
base_eta_T = r0['eta_T']
pnames  = ['T [K]', 'p_H_total [Pa]', 'δ [µm]',
           'n_tubes', 'L [m]', 'p_perm [Pa]', 'x_H_in']
lo_pars = [{'T': 573}, {'p_H_total': 10},  {'delta': 200e-6},
           {'n_tubes': 5}, {'L': 0.25}, {'p_perm': 5.0}, {'x_H_in': 0.1}]
hi_pars = [{'T': 723}, {'p_H_total': 500}, {'delta': 50e-6},
           {'n_tubes': 50}, {'L': 1.0}, {'p_perm': 0.0}, {'x_H_in': 0.9}]
eta_lo  = [solve_permeator(q)['eta_T'] for q in lo_pars]
eta_hi  = [solve_permeator(q)['eta_T'] for q in hi_pars]
plot_tornado(pnames, eta_lo, base_eta_T, eta_hi, OUT + 'tornado.png')

# ── Geometry sensitivity (2 figures) ─────────────────────────────────────────
print("\nRunning geometry sweep...")
n_vals = [100,1000,10_000]
r_n    = [solve_permeator({'n_tubes': n}) for n in n_vals]
plot_geometry_pressure_profile(r_n, n_vals, 'n_tubes',
                               OUT + 'geometry_pressure_profile.png')
plot_geometry_species_profiles(r_n, n_vals, 'n_tubes',
                               OUT + 'geometry_species_profiles.png')

# ── Tubes vs partial pressure contour map (1 figure) ─────────────────────────
print("\nRunning tubes vs partial pressure sweep...")
n_tubes_arr = np.array([10, 50, 100, 500, 1000, 5000, 10000, 50000])
p_H_arr     = np.geomspace(0.1, 100, 12)

def _eta_total(r):
    p    = r['params']
    F_in = p['F_H2_in'] + p['F_T2_in']
    F_out = r['F_H2'][-1] + r['F_T2'][-1]
    return (F_in - F_out) / F_in if F_in > 0 else 0.0

eta_grid = [
    [_eta_total(solve_permeator({'n_tubes': int(n), 'p_H_total': float(pH)}))
     for n in n_tubes_arr]
    for pH in p_H_arr
]
plot_tubes_vs_pressure_map(n_tubes_arr, p_H_arr, eta_grid,
                           OUT + 'tubes_vs_pressure_map.png')

print(f"\nDone — 17 plots saved to {OUT}")