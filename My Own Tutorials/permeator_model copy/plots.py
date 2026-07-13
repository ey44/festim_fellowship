"""
plots.py
========
Individual single-panel figures for the H2/T2 coupled permeator model.
Every function produces exactly one figure saved as a separate PNG at dpi=200.

Output files (16 total)
-----------------------
axial_molar_flows.png
axial_partial_pressures.png
axial_flux.png
axial_lambda.png
recovery_bars.png
resistance_split.png
temp_recovery.png
temp_lambda.png
pH_recovery.png
pperm_recovery.png
composition_map.png
regime_lambda_profile.png
regime_composition.png
tornado.png
geometry_pressure_profile.png
geometry_species_profiles.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os

# ── Style ─────────────────────────────────────────────────────────────────────
DPI    = 200
BLUE   = '#1f77b4'
RED    = '#d62728'
GREEN  = '#2ca02c'
ORANGE = '#ff7f0e'
PURP   = '#9467bd'
GREY   = '#7f7f7f'
LBLUE  = '#aec7e8'
LRED   = '#ffb3b3'
LGREEN = '#b2dfb2'

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.labelsize':    12,
    'axes.titlesize':    13,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'legend.fontsize':   10,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linewidth':    0.6,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        DPI,
})


def _save(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {os.path.basename(path)}")


def _regime_label(lam):
    if lam > 100:    return 'surface-limited'
    elif lam > 0.01: return 'intermediate'
    else:            return 'diffusion-limited'


def _header(p):
    """Short parameter string for figure titles."""
    return (f"T={p['T']:.0f} K  |  "
            f"P_H={p['p_H_total']:.0f} Pa  |  "
            f"x_H={p['x_H_in']:.2f}  |  "
            f"n={p['n_tubes']}  |  "
            f"δ={p['delta']*1e6:.0f} µm")


# ══════════════════════════════════════════════════════════════════════════════
# AXIAL PROFILE PLOTS  (4 separate figures from one result dict)
# ══════════════════════════════════════════════════════════════════════════════

def plot_axial_molar_flows(result, path):
    """Atomic molar flows N_H and N_T along the tube."""
    p      = result['params']
    z_norm = result['z'] / p['L']
    N_H    = result['N'][0]
    N_T    = result['N'][1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z_norm, N_H * 1e3, color=BLUE, lw=2.2, label='N_H  (H atoms → H₂)')
    ax.plot(z_norm, N_T * 1e3, color=RED,  lw=2.2, label='N_T  (T atoms → T₂)')

    ax.set_xlabel('z / L  (normalised axial position)')
    ax.set_ylabel('Atomic molar flow [mmol/s]')
    ax.set_title(f'Atomic molar flows along tube\n{_header(p)}')
    ax.legend()
    ax.set_xlim(0, 1)

    ax.text(0.97, 0.55,
            f'η_H = {100*result["eta_H"]:.2f}%\nη_T = {100*result["eta_T"]:.2f}%',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor=GREY, alpha=0.9))
    fig.tight_layout()
    _save(fig, path)


def plot_axial_partial_pressures(result, path):
    """Partial pressures p_H2, p_T2, and P_H_total along the tube."""
    p      = result['params']
    z_norm = result['z'] / p['L']
    P_H    = result['P_H_total']
    x_H    = result['x_H']
    x_T    = result['x_T']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z_norm, P_H,        color=GREY,   lw=1.5, ls='--',
            label='P_H,total', alpha=0.8)
    ax.plot(z_norm, x_H * P_H,  color=BLUE,   lw=2.2, label='p_H₂')
    ax.plot(z_norm, x_T * P_H,  color=RED,    lw=2.2, label='p_T₂')
    ax.axhline(p['p_perm'], color='k', lw=1.0, ls=':',
               alpha=0.6, label=f'p_perm = {p["p_perm"]:.1f} Pa')

    ax.set_xlabel('z / L')
    ax.set_ylabel('Partial pressure [Pa]')
    ax.set_title(f'Partial pressures along tube\n{_header(p)}')
    ax.legend()
    ax.set_xlim(0, 1)
    fig.tight_layout()
    _save(fig, path)


def plot_axial_flux(result, path):
    """Local atomic permeation flux J_H and J_T along the tube."""
    p      = result['params']
    z_norm = result['z'] / p['L']
    J_H    = result['J_H2']
    J_T    = result['J_T2']
    d      = result['diagnostics']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z_norm, J_H * 1e3, color=BLUE, lw=2.2,
            label='J_H₂  [mmol H₂/m²/s]')
    ax.plot(z_norm, J_T * 1e3, color=RED,  lw=2.2,
            label='J_T₂  [mmol T₂/m²/s]')

    ax.set_xlabel('z / L')
    ax.set_ylabel('Molecular permeation flux [mmol/m²/s]')
    ax.set_title(f'Local permeation flux along tube\n{_header(p)}')
    ax.legend()
    ax.set_xlim(0, 1)

    if d['back_diffusion_H2']:
        ax.text(0.5, 0.96, '⚠  H₂ back-diffusion detected',
                transform=ax.transAxes, ha='center', va='top',
                color='darkred', fontsize=10, fontweight='bold')
    if d['back_diffusion_T2']:
        ax.text(0.5, 0.88, '⚠  T₂ back-diffusion detected',
                transform=ax.transAxes, ha='center', va='top',
                color='darkred', fontsize=10, fontweight='bold')
    fig.tight_layout()
    _save(fig, path)


def plot_axial_lambda(result, path):
    """Regime parameter Λ(z) along the tube with shaded regime regions."""
    p      = result['params']
    z_norm = result['z'] / p['L']
    Lam    = result['Lambda']

    fig, ax = plt.subplots(figsize=(8, 5))

    # Regime shading
    ax.axhspan(100,  1e14,  alpha=0.10, color=RED,    label='Surface-limited  (Λ > 100)')
    ax.axhspan(0.01, 100,   alpha=0.10, color=ORANGE, label='Intermediate  (0.01 < Λ < 100)')
    ax.axhspan(1e-6, 0.01,  alpha=0.10, color=GREEN,  label='Diffusion-limited  (Λ < 0.01)')

    ax.semilogy(z_norm, Lam, color=PURP, lw=2.5, zorder=5, label='Λ(z)')
    ax.axhline(100,   color=RED,   lw=0.9, ls='--', alpha=0.5)
    ax.axhline(1.0,   color=GREY,  lw=0.9, ls='--', alpha=0.5)
    ax.axhline(0.01,  color=GREEN, lw=0.9, ls='--', alpha=0.5)

    ax.set_xlabel('z / L')
    ax.set_ylabel('Λ  (log scale)')
    ax.set_title(f'Regime diagnostic Λ(z)\n{_header(p)}')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_ylim(max(1e-4, Lam.min() * 0.3), Lam.max() * 3)

    ax.text(0.02, 0.97,
            f'Inlet: Λ = {Lam[0]:.2e}  →  {_regime_label(Lam[0])}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.text(0.98, 0.08,
            f'Exit:  Λ = {Lam[-1]:.2e}  →  {_regime_label(Lam[-1])}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    fig.tight_layout()
    _save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# RECOVERY AND RESISTANCE
# ══════════════════════════════════════════════════════════════════════════════

def plot_recovery_bars(result, path):
    """Bar chart of η_H and η_T recovery fractions."""
    p   = result['params']
    eH  = result['eta_H']
    eT  = result['eta_T']

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(['H₂  (η_H)', 'T₂  (η_T)'],
                  [eH * 100, eT * 100],
                  color=[BLUE, RED], width=0.45, alpha=0.85,
                  edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, [eH, eT]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val * 100 + 0.5,
                f'{val*100:.2f}%',
                ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    ax.set_ylim(0, 115)
    ax.set_ylabel('Recovery η [%]')
    ax.set_title(f'Species recovery\n{_header(p)}')
    ax.text(0.97, 0.97,
            f'η_T (tritium atoms) = {100*eT:.2f}%',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    fig.tight_layout()
    _save(fig, path)


def plot_resistance_split(result, path):
    """Horizontal bar showing R1 (gas film) vs R2 (surface) split."""
    p  = result['params']
    d  = result['diagnostics']
    R1 = d['R1_gas_film']
    R2 = d['R2_surface_H']
    Rt = R1 + R2

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.barh([1, 0],
            [100 * R1 / Rt, 100 * R2 / Rt],
            color=[GREEN, ORANGE], alpha=0.85, height=0.45,
            edgecolor='white', linewidth=1.2)

    ax.set_yticks([1, 0])
    ax.set_yticklabels(['Gas film\n(R₁ = RT/k_m)',
                        'Surface\n(R₂ = 1/K_a)'], fontsize=11)
    ax.set_xlabel('Fraction of total feed-side resistance [%]')
    ax.set_title(f'Feed-side resistance split\n{_header(p)}')
    ax.set_xlim(0, 115)

    for yp, R, val in zip([1, 0], [R1, R2],
                           [100*R1/Rt, 100*R2/Rt]):
        ax.text(val + 1, yp,
                f'{val:.1f}%\n({R:.2e} m²·s·Pa/mol)',
                va='center', fontsize=9)

    ax.text(0.97, 0.05,
            f"Re = {d['Re']:.0f}  ({d['flow_regime']})\n"
            f"Sh = {d['Sh']:.2f}   k_m = {d['k_m']:.3e} m/s",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor=GREY, alpha=0.9))
    fig.tight_layout()
    _save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# TEMPERATURE SENSITIVITY  (2 separate figures)
# ══════════════════════════════════════════════════════════════════════════════

def plot_temp_recovery(T_arr, results, path):
    """η_H and η_T vs temperature."""
    T_C   = np.array(T_arr) - 273.15
    eta_H = np.array([r['eta_H'] for r in results]) * 100
    eta_T = np.array([r['eta_T'] for r in results]) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T_C, eta_H, 'o-', color=BLUE, lw=2, ms=7, label='η_H  (H₂)')
    ax.plot(T_C, eta_T, 's-', color=RED,  lw=2, ms=7, label='η_T  (T₂)')
    ax.set_xlabel('Temperature [°C]')
    ax.set_ylabel('Recovery η [%]')
    ax.set_title('Recovery vs temperature')
    ax.legend()
    ax.set_xlim(T_C[0] - 10, T_C[-1] + 10)
    fig.tight_layout()
    _save(fig, path)


def plot_temp_lambda(T_arr, results, path):
    """Inlet Λ vs temperature with regime shading."""
    T_C     = np.array(T_arr) - 273.15
    lam_in  = np.array([r['diagnostics']['Lambda_inlet'] for r in results])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(T_C, lam_in, 'D-', color=PURP, lw=2, ms=7, label='Λ_inlet')
    ax.axhline(100,  color=RED,   lw=1, ls='--', alpha=0.6, label='Λ = 100')
    ax.axhline(1.0,  color=GREY,  lw=1, ls='--', alpha=0.6, label='Λ = 1')
    ax.axhline(0.01, color=GREEN, lw=1, ls='--', alpha=0.6, label='Λ = 0.01')

    ylo = lam_in.min() * 0.3
    yhi = lam_in.max() * 3
    ax.set_ylim(ylo, yhi)
    ax.axhspan(100, yhi,  alpha=0.08, color=RED,    zorder=0)
    ax.axhspan(0.01, 100, alpha=0.08, color=ORANGE, zorder=0)
    ax.axhspan(ylo, 0.01, alpha=0.08, color=GREEN,  zorder=0)

    ax.set_xlabel('Temperature [°C]')
    ax.set_ylabel('Λ_inlet  (log scale)')
    ax.set_title('Regime parameter Λ vs temperature')
    ax.legend(fontsize=9)
    ax.set_xlim(T_C[0] - 10, T_C[-1] + 10)
    fig.tight_layout()
    _save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# PRESSURE SENSITIVITY  (2 separate figures)
# ══════════════════════════════════════════════════════════════════════════════

def plot_pH_recovery(p_H_arr, results, path):
    """Recovery vs inlet hydrogen partial pressure (log x-axis)."""
    eta_H = np.array([r['eta_H'] for r in results]) * 100
    eta_T = np.array([r['eta_T'] for r in results]) * 100
    lam   = np.array([r['diagnostics']['Lambda_inlet'] for r in results])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()

    l1, = ax.semilogx(p_H_arr, eta_H, 'o-', color=BLUE,  lw=2, ms=7,
                      label='η_H')
    l2, = ax.semilogx(p_H_arr, eta_T, 's-', color=RED,   lw=2, ms=7,
                      label='η_T')
    l3, = ax2.semilogx(p_H_arr, lam,  'D--', color=PURP, lw=1.5, ms=5,
                       alpha=0.7, label='Λ_inlet (right)')

    ax.set_xlabel('P_H,total at inlet [Pa]')
    ax.set_ylabel('Recovery η [%]')
    ax2.set_ylabel('Λ_inlet', color=PURP)
    ax2.tick_params(axis='y', colors=PURP)
    ax2.set_yscale('log')
    ax.set_title('Recovery vs inlet hydrogen partial pressure')
    ax.legend([l1, l2, l3], [l.get_label() for l in [l1, l2, l3]],
              fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def plot_pperm_recovery(p_perm_arr, results, path):
    """Recovery vs permeate back-pressure."""
    eta_H = np.array([r['eta_H'] for r in results]) * 100
    eta_T = np.array([r['eta_T'] for r in results]) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_perm_arr, eta_H, 'o-', color=BLUE, lw=2, ms=7, label='η_H')
    ax.plot(p_perm_arr, eta_T, 's-', color=RED,  lw=2, ms=7, label='η_T')
    ax.set_xlabel('Permeate back-pressure p_perm [Pa]')
    ax.set_ylabel('Recovery η [%]')
    ax.set_title('Recovery vs permeate back-pressure')
    ax.legend()

    # Mark back-pressure = 50% of driving force
    p_H = results[0]['params']['p_H_total']
    bp50 = p_H * 0.25   # sqrt(p_perm) = 0.5*sqrt(p_H) -> p_perm = 0.25*p_H
    if p_perm_arr[0] < bp50 < p_perm_arr[-1]:
        ax.axvline(bp50, color=GREY, lw=1, ls=':', alpha=0.7)
        ax.text(bp50 * 1.05, ax.get_ylim()[0] * 1.05 + 1,
                '50% DF\nlost', fontsize=8, color=GREY)
    fig.tight_layout()
    _save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITION MAP  (1 figure)
# ══════════════════════════════════════════════════════════════════════════════

def plot_composition_map(x_H_arr, T_arr, eta_T_grid, path):
    """2D contour map of η_T vs (x_H_in, T)."""
    T_C    = np.array(T_arr) - 273.15
    Z      = np.array(eta_T_grid) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    levels  = np.linspace(0, 100, 21)
    cf = ax.contourf(x_H_arr, T_C, Z, levels=levels, cmap='RdYlGn')
    cs = ax.contour(x_H_arr, T_C, Z,
                    levels=[20, 40, 60, 80, 95],
                    colors='white', linewidths=0.9, alpha=0.75)
    ax.clabel(cs, fmt='%d%%', fontsize=9)

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label('η_T  [%]', fontsize=12)
    cbar.set_ticks(np.arange(0, 101, 20))

    ax.axvline(0.5, color='white', lw=1.5, ls='--', alpha=0.8)
    ax.text(0.51, T_C[-1] - 10, 'x_H = 0.5',
            color='white', fontsize=9, fontweight='bold')

    ax.set_xlabel('Inlet H atomic fraction  x_H')
    ax.set_ylabel('Temperature [°C]')
    ax.set_title('Tritium (T) recovery map\nη_T vs composition and temperature')
    fig.tight_layout()
    _save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# REGIME MAP  (2 separate figures)
# ══════════════════════════════════════════════════════════════════════════════

def plot_regime_lambda_profile(result, path):
    """Λ(z) profile with regime shading — single panel."""
    p      = result['params']
    z_norm = result['z'] / p['L']
    Lam    = result['Lambda']

    fig, ax = plt.subplots(figsize=(8, 5))
    ylo = max(1e-4, Lam.min() * 0.3)
    yhi = Lam.max() * 3

    ax.axhspan(100,  yhi,  alpha=0.10, color=RED,    label='Surface-limited  (Λ > 100)')
    ax.axhspan(0.01, 100,  alpha=0.10, color=ORANGE, label='Intermediate')
    ax.axhspan(ylo,  0.01, alpha=0.10, color=GREEN,  label='Diffusion-limited  (Λ < 0.01)')

    ax.semilogy(z_norm, Lam, color=PURP, lw=2.5, zorder=5, label='Λ(z)')
    ax.axhline(100,  color=RED,   lw=0.9, ls='--', alpha=0.5)
    ax.axhline(1.0,  color=GREY,  lw=0.9, ls='--', alpha=0.5)
    ax.axhline(0.01, color=GREEN, lw=0.9, ls='--', alpha=0.5)

    ax.set_xlabel('z / L')
    ax.set_ylabel('Λ  (log scale)')
    ax.set_title(f'Regime parameter Λ along tube\n{_header(p)}')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_ylim(ylo, yhi)

    ax.text(0.02, 0.97,
            f'Inlet: Λ = {Lam[0]:.2e}  →  {_regime_label(Lam[0])}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.text(0.98, 0.08,
            f'Exit:  Λ = {Lam[-1]:.2e}  →  {_regime_label(Lam[-1])}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    fig.tight_layout()
    _save(fig, path)


def plot_regime_composition(result, path):
    """x_H(z) and P_H_total(z) along the tube."""
    p      = result['params']
    z_norm = result['z'] / p['L']
    P_H    = result['P_H_total']
    x_H    = result['x_H']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()

    ax.plot(z_norm, P_H,         color=GREY,  lw=1.5, ls='--',
            label='P_H,total', alpha=0.8)
    ax.plot(z_norm, x_H * P_H,   color=BLUE,  lw=2.2, label='p_H₂')
    ax.plot(z_norm, (1-x_H)*P_H, color=RED,   lw=2.2, label='p_T₂')
    ax2.plot(z_norm, x_H,        color=GREEN, lw=1.5, ls=':',
             label='x_H (right)')

    ax.set_xlabel('z / L')
    ax.set_ylabel('Partial pressure [Pa]')
    ax2.set_ylabel('x_H', color=GREEN)
    ax2.tick_params(axis='y', colors=GREEN)
    ax.set_title(f'Composition and pressure along tube\n{_header(p)}')

    lines, labels = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lb2, fontsize=9)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    _save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# TORNADO  (1 figure)
# ══════════════════════════════════════════════════════════════════════════════

def plot_tornado(param_names, eta_T_low, eta_T_base, eta_T_high, path,
                 title='Sensitivity of η_T to individual parameters'):
    """Horizontal tornado chart of η_T sensitivity."""
    base  = eta_T_base * 100
    lo    = np.array(eta_T_low)  * 100 - base
    hi    = np.array(eta_T_high) * 100 - base
    swing = np.abs(hi - lo)
    order = np.argsort(swing)
    names = [param_names[i] for i in order]
    lo    = lo[order]
    hi    = hi[order]
    n     = len(names)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.55 * n + 2)))

    for i, (l, h) in enumerate(zip(lo, hi)):
        ax.barh(i, l, left=0, color=LRED,  height=0.55,
                edgecolor='white', linewidth=0.8)
        ax.barh(i, h, left=0, color=LBLUE, height=0.55,
                edgecolor='white', linewidth=0.8)
        ax.text(l - 0.3, i, f'{l:+.1f} pp', ha='right',  va='center', fontsize=9)
        ax.text(h + 0.3, i, f'{h:+.1f} pp', ha='left',   va='center', fontsize=9)

    ax.axvline(0, color='black', lw=1.5)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(names)
    ax.set_xlabel(f'Change in η_T relative to base case ({base:.1f}%)  [percentage points]')
    ax.set_title(title)

    ax.legend(handles=[Patch(facecolor=LRED,  label='Low setting'),
                        Patch(facecolor=LBLUE, label='High setting')],
              fontsize=9, loc='lower right')
    fig.tight_layout()
    _save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY SENSITIVITY  (2 separate figures)
# ══════════════════════════════════════════════════════════════════════════════

def plot_geometry_pressure_profile(results, param_vals, param_label, path):
    """P_H_total(z) overlay for different geometry parameter values."""
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(param_vals)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for r, val, col in zip(results, param_vals, colors):
        p      = r['params']
        z_norm = r['z'] / p['L']
        eta_T  = r['eta_T']
        ax.plot(z_norm, r['P_H_total'], color=col, lw=2,
                label=f'{param_label} = {val}  (η_T = {100*eta_T:.1f}%)')

    ax.set_xlabel('z / L')
    ax.set_ylabel('P_H,total [Pa]')
    ax.set_title(f'Total hydrogen pressure profiles\nvarying {param_label}')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    _save(fig, path)


def plot_geometry_species_profiles(results, param_vals, param_label, path):
    """p_H2(z) and p_T2(z) overlay for different geometry parameter values."""
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(param_vals)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for r, val, col in zip(results, param_vals, colors):
        p      = r['params']
        z_norm = r['z'] / p['L']
        P_H    = r['P_H_total']
        ax.plot(z_norm, r['x_H'] * P_H,       color=col, lw=2.0, ls='-')
        ax.plot(z_norm, (1 - r['x_H']) * P_H, color=col, lw=2.0, ls='--')

    ax.set_xlabel('z / L')
    ax.set_ylabel('Partial pressure [Pa]')
    ax.set_title(f'Species partial pressure profiles\nvarying {param_label}')

    # Legend: colour = param value, style = species
    handles = (
        [Line2D([0], [0], color=c, lw=2)
         for c in colors]
        + [Line2D([0], [0], color='k', lw=2, ls='-',  label='p_H₂'),
           Line2D([0], [0], color='k', lw=2, ls='--', label='p_T₂')]
    )
    labels = (
        [f'{param_label} = {v}' for v in param_vals]
        + ['p_H₂', 'p_T₂']
    )
    ax.legend(handles, labels, fontsize=8, ncol=2)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    _save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# TUBES vs PARTIAL PRESSURE CONTOUR MAP
# ══════════════════════════════════════════════════════════════════════════════

def plot_tubes_vs_pressure_map(n_tubes_arr, p_H_arr, eta_grid, path):
    """
    2D contour map of total-stream recovery η_total vs (n_tubes, p_H_total).

    Parameters
    ----------
    n_tubes_arr : 1D array of n_tubes values (x-axis, log scale)
    p_H_arr     : 1D array of p_H_total values in Pa (y-axis, log scale)
    eta_grid    : 2D array shape (len(p_H_arr), len(n_tubes_arr)) of η_total
    target_eta  : recovery contour to highlight (default 0.80)
    """
    Z = np.array(eta_grid) * 100   # percent

    fig, ax = plt.subplots(figsize=(9, 6))

    levels = np.linspace(0, 100, 21)
    cf = ax.contourf(n_tubes_arr, p_H_arr, Z,
                     levels=levels, cmap='RdYlGn')

    # Labelled iso-recovery contours
    iso_levels = [20, 40, 60, 80, 90, 95]
    cs = ax.contour(n_tubes_arr, p_H_arr, Z,
                    levels=iso_levels,
                    colors='white', linewidths=0.9, alpha=0.75)
    ax.clabel(cs, fmt='%d%%', fontsize=9)

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label('η_total  [%]', fontsize=12)
    cbar.set_ticks(np.arange(0, 101, 20))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of tubes  n_tubes', fontsize=12)
    ax.set_ylabel('Inlet H partial pressure  p_H,total [Pa]', fontsize=12)
    ax.set_title('Total-stream recovery  η_total\nvs number of tubes and inlet partial pressure')

    fig.tight_layout()
    _save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST / DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from permeator_v2 import solve_permeator

    OUT = '/mnt/user-data/outputs/plots/'
    os.makedirs(OUT, exist_ok=True)
    print("Generating all individual plots...\n")

    # ── Base result ───────────────────────────────────────────────────────────
    r0 = solve_permeator()

    # ── Axial profiles (4 figures) ────────────────────────────────────────────
    plot_axial_molar_flows(      r0, OUT + 'axial_molar_flows.png')
    plot_axial_partial_pressures(r0, OUT + 'axial_partial_pressures.png')
    plot_axial_flux(             r0, OUT + 'axial_flux.png')
    plot_axial_lambda(           r0, OUT + 'axial_lambda.png')

    # ── Recovery & resistance (2 figures) ────────────────────────────────────
    plot_recovery_bars(   r0, OUT + 'recovery_bars.png')
    plot_resistance_split(r0, OUT + 'resistance_split.png')

    # ── Temperature sensitivity (2 figures) ───────────────────────────────────
    T_arr = [473, 523, 573, 623, 673, 723]
    r_T   = [solve_permeator({'T': T}) for T in T_arr]
    plot_temp_recovery(T_arr, r_T, OUT + 'temp_recovery.png')
    plot_temp_lambda(  T_arr, r_T, OUT + 'temp_lambda.png')

    # ── Pressure sensitivity (2 figures) ─────────────────────────────────────
    pH_arr = [5, 10, 25, 50, 100, 250, 500, 1000]
    r_pH   = [solve_permeator({'p_H_total': p}) for p in pH_arr]
    plot_pH_recovery(pH_arr, r_pH, OUT + 'pH_recovery.png')

    pp_arr = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    r_pp   = [solve_permeator({'p_perm': p}) for p in pp_arr]
    plot_pperm_recovery(pp_arr, r_pp, OUT + 'pperm_recovery.png')

    # ── Composition map (1 figure) ────────────────────────────────────────────
    xH_arr   = np.linspace(0.05, 0.95, 12)
    T_map    = [473, 523, 573, 623, 673, 723]
    eta_grid = [
        [solve_permeator({'T': T, 'x_H_in': float(xH)})['eta_T']
         for xH in xH_arr]
        for T in T_map
    ]
    plot_composition_map(xH_arr, T_map, eta_grid, OUT + 'composition_map.png')

    # ── Regime map (2 figures) ────────────────────────────────────────────────
    plot_regime_lambda_profile(r0, OUT + 'regime_lambda_profile.png')
    plot_regime_composition(   r0, OUT + 'regime_composition.png')

    # ── Tornado (1 figure) ────────────────────────────────────────────────────
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

    # ── Geometry sensitivity (2 figures) ─────────────────────────────────────
    n_vals = [1, 5, 10, 50, 100]
    r_n    = [solve_permeator({'n_tubes': n}) for n in n_vals]
    plot_geometry_pressure_profile(r_n, n_vals, 'n_tubes',
                                   OUT + 'geometry_pressure_profile.png')
    plot_geometry_species_profiles(r_n, n_vals, 'n_tubes',
                                   OUT + 'geometry_species_profiles.png')

    print(f"\nDone — 16 plots saved to {OUT}")
