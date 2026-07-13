"""
hx_plots.py
===========
Visualisation of the HX tritium-permeation model (Phases 0-3), using the
shared plot_utils conventions (one plot per figure).

Figures (saved to ./outputs/):
  1. hx_temperature_profiles.png   T_He, T_sec, T_wall vs z
  2. hx_flux_profile.png           local J_T2(z), log scale
  3. hx_cumulative_permeation.png  cumulative fraction of Q_T vs z
  4. hx_flux_vs_Twall.png          J vs T_wall for several p_T2
  5. hx_flux_map_contour.png       contour of J over (T_wall, p_T2)
  6. hx_QT_vs_pT2.png              total Q_T vs p_T2 for f_phi = 1/0.1/0.01
  7. hx_QT_vs_U_area.png           total Q_T vs U (area trade-off)

Usage:
  python3 hx_plots.py            # light theme
  python3 hx_plots.py --brand    # Type One Energy dark theme
"""

import sys
import warnings
import numpy as np
import matplotlib as mpl

from plot_utils import make_fig, make_contour_fig, save_fig, COLORS, BRAND_COLORS
from hx_thermal_sizing import size_hx
from hx_1d_model import compute_flux_analytical
from axial_integration import integrate_hx

THEME  = 'brand' if '--brand' in sys.argv else 'light'
FOLDER = 'outputs'
C = BRAND_COLORS if THEME == 'brand' else {
    'blue_deep': COLORS['blue'], 'orange': COLORS['orange'],
    'blue_light': COLORS['green'], 'blue_mid': COLORS['purple'],
    'grey': COLORS['grey'],
}

warnings.simplefilter('ignore')   # T-range extrapolation warned elsewhere

import os
os.makedirs(FOLDER, exist_ok=True)

# ── Shared data ──────────────────────────────────────────────────────────────
design, prof = size_hx(verbose=False)
base = integrate_hx(quiet=True)
z, L = prof['z_m'], design['L_m']

# ── 1. Temperature profiles ──────────────────────────────────────────────────
fig, ax = make_fig(
    xlabel='Axial position z (m)  —  z=0 is He inlet (hot end)',
    ylabel='Temperature (°C)',
    title='Counterflow temperature profiles (Phase 0)',
    theme=THEME,
)
ax.plot(z, prof['T_he_K'] - 273.15,   '-',  color=C['orange'],
        label='He (600→300 °C)')
ax.plot(z, prof['T_sec_K'] - 273.15,  '-',  color=C['blue_deep'],
        label='Steam (290→580 °C)')
ax.plot(z, prof['T_wall_he_K'] - 273.15, '--', color=C['grey'],
        label='Wall, He-side surface')
ax.legend(fontsize=9)
ax.annotate('wall sits near steam T:\nHe film ≈ 93% of resistance',
            xy=(0.45 * L, 430), fontsize=8.5,
            color='white' if THEME == 'brand' else '#444444')
save_fig(fig, 'hx_temperature_profiles.png', folder=FOLDER)

# ── 2. Local flux profile ────────────────────────────────────────────────────
fig, ax = make_fig(
    xlabel='Axial position z (m)',
    ylabel=r'Local permeation flux $J_{T_2}$ (mol m$^{-2}$ s$^{-1}$)',
    title='Local tritium flux along the HX (baseline, 10 Pa, bare wall)',
    theme=THEME, log_y=True,
)
ax.plot(base['z'], base['J_profile'], '-', color=C['orange'])
save_fig(fig, 'hx_flux_profile.png', folder=FOLDER)

# ── 3. Cumulative permeation ─────────────────────────────────────────────────
trapz = getattr(np, 'trapezoid', None) or np.trapz
J = base['J_profile']
cum = np.array([trapz(J[:i + 1], base['z'][:i + 1]) for i in range(len(J))])
cum /= cum[-1]

fig, ax = make_fig(
    xlabel='Fraction of HX length from He inlet',
    ylabel='Cumulative fraction of total permeation',
    title='Where the tritium permeates (baseline)',
    theme=THEME,
)
ax.plot(base['z'] / L, cum, '-', color=C['blue_deep'])
for fr, ls in ((0.5, ':'), (0.9, '--')):
    i = np.searchsorted(cum, fr)
    ax.axhline(fr, ls=ls, lw=0.8, color=C['grey'], alpha=0.6)
    ax.axvline(base['z'][i] / L, ls=ls, lw=0.8, color=C['grey'], alpha=0.6)
    ax.annotate(f"{fr:.0%} in first {base['z'][i]/L:.0%}",
                xy=(base['z'][i] / L + 0.02, fr - 0.05), fontsize=8.5,
                color='white' if THEME == 'brand' else '#444444')
save_fig(fig, 'hx_cumulative_permeation.png', folder=FOLDER)

# ── 4. J vs T_wall for several p_T2 ──────────────────────────────────────────
T_grid = np.linspace(280, 600, 80)
fig, ax = make_fig(
    xlabel='Wall temperature (°C)',
    ylabel=r'$J_{T_2}$ (mol m$^{-2}$ s$^{-1}$)',
    title='Local flux vs wall temperature (Phase 2 flux map)',
    theme=THEME, log_y=True,
)
for p, col in zip((1.0, 10.0, 100.0),
                  (C['blue_light'], C['orange'], C['blue_deep'])):
    Jt = [compute_flux_analytical(T + 273.15, p)['J_T2'] for T in T_grid]
    ax.plot(T_grid, Jt, '-', color=col, label=f'$p_{{T_2}}$ = {p:.0f} Pa')
ax.axvspan(prof['T_wall_he_K'].min() - 273.15,
           prof['T_wall_he_K'].max() - 273.15,
           alpha=0.10, color=C['grey'], label='HX wall T range')
ax.legend(fontsize=9)
save_fig(fig, 'hx_flux_vs_Twall.png', folder=FOLDER)

# ── 5. Flux-map contour J(T_wall, p_T2) ──────────────────────────────────────
p_grid = np.logspace(0, 2, 40)
TT, PP = np.meshgrid(T_grid, p_grid)
ZZ = np.array([[compute_flux_analytical(T + 273.15, p)['J_T2']
                for T in T_grid] for p in p_grid])
fig, ax = make_contour_fig(
    TT, PP, np.log10(ZZ),
    xlabel='Wall temperature (°C)',
    ylabel=r'$p_{T_2}$ (Pa)',
    clabel=r'log$_{10}$ $J_{T_2}$ (mol m$^{-2}$ s$^{-1}$)',
    title='Phase-2 flux map (bare Inconel 617)',
    log_y=True, theme=THEME, figsize=(7, 5),
    contour_lines=[-11, -10, -9, -8],
    contour_fmt='{:.0f}',
)
save_fig(fig, 'hx_flux_map_contour.png', folder=FOLDER)

# ── 6. Total Q_T vs p_T2 for oxide states ────────────────────────────────────
p_vals = np.logspace(0, 2, 9)
fig, ax = make_fig(
    xlabel=r'Inlet $p_{T_2}$ (Pa)',
    ylabel='Tritium permeation (Ci/day)',
    title='Total permeation vs tritium partial pressure',
    theme=THEME, log_x=True, log_y=True,
)
for f, col, lab in ((1.0, C['orange'], 'bare metal'),
                    (0.1, C['blue_light'], r'$f_\Phi$ = 0.1'),
                    (0.01, C['blue_deep'], r'oxidised ($f_\Phi$ = 0.01)')):
    ci = [integrate_hx(p_T2_in=p, f_phi=f, quiet=True)['Ci_per_day']
          for p in p_vals]
    ax.plot(p_vals, ci, 'o-', color=col, label=lab)
ax.axvline(10.0, ls=':', lw=0.8, color=C['grey'])
ax.annotate('baseline\n10 Pa', xy=(10.5, ax.get_ylim()[0] * 2), fontsize=8.5,
            color='white' if THEME == 'brand' else '#444444')
ax.legend(fontsize=9)
save_fig(fig, 'hx_QT_vs_pT2.png', folder=FOLDER)

# ── 7. Q_T vs U (area trade-off) ─────────────────────────────────────────────
configs = [(2000.0, 2000.0), (2000.0, 4000.0), (3000.0, 2000.0),
           (3000.0, 4000.0), (3000.0, 8000.0), (5000.0, 4000.0),
           (5000.0, 8000.0), (8000.0, 4000.0), (8000.0, 8000.0)]
U_l, Ci_l, A_l = [], [], []
for re_t, hs in configs:
    r = integrate_hx(design_kw=dict(Re_target=re_t, h_sec=hs), quiet=True)
    U_l.append(r['U_W_m2K']); Ci_l.append(r['Ci_per_day']); A_l.append(r['A_m2'])

fig, ax = make_fig(
    xlabel=r'Overall heat-transfer coefficient U (W m$^{-2}$ K$^{-1}$)',
    ylabel='Tritium permeation (Ci/day)',
    title=r'Permeation vs U: better U $\rightarrow$ less area $\rightarrow$ less tritium',
    theme=THEME,
)
order = np.argsort(U_l)
U_s = np.array(U_l)[order]; Ci_s = np.array(Ci_l)[order]; A_s = np.array(A_l)[order]
ax.plot(U_s, Ci_s, 'o-', color=C['orange'])
for u, c_, a in zip(U_s[::2], Ci_s[::2], A_s[::2]):
    ax.annotate(f'{a:.0f} m²', xy=(u, c_), xytext=(4, 6),
                textcoords='offset points', fontsize=8,
                color='white' if THEME == 'brand' else '#444444')
save_fig(fig, 'hx_QT_vs_U_area.png', folder=FOLDER)

print(f"\nAll figures in ./{FOLDER}/ (theme: {THEME})")
