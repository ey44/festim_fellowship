"""
pipe_resistance_plot.py
=======================
Diagnostic: compare the three transport resistances for tritium permeation
through SS316L piping as a function of temperature.

Four curves (all in √p-space, area-specific to inner surface):

  1. Wall – flat plate      R = δ / Φ(T)
  2. Wall – cylindrical     R = r_i · ln(r_o/r_i) / Φ(T)
  3. Surface adsorption     R = 1 / (2 · K_a(T) · √p_ref)   [Pd/Ag, lower bound]
  4. Gas-film (He carrier)  R = RT / (2 · k_m(T) · √p_ref)

All units: m²·s·Pa⁰·⁵/mol

The factor of 2 on the gas-film and surface terms comes from linearising the
quadratic (in √p) terms around p_ref: for J = K_a·p, dJ/d(√p)|_{√p_ref} = 2·K_a·√p_ref.

The K_a used here is for clean Pd/Ag (Vadrucci 2013) — the most favourable
surface possible. SS316L surface resistance is unknown but almost certainly
higher, so the plotted surface curve is a lower bound.

Pipe geometry used: 20 mm OD, 2 mm wall → r_i = 8 mm, r_o = 10 mm.
He flow: 5 Nl/min, 1 bar, single pipe.
Reference pressure: p_ref = 100 Pa (representative HT partial pressure).
"""

import numpy as np
import warnings

from material_library import get_material, permeability_species, M_T
from transport import surface_resistance_H, mass_transfer_coeff
from plot_utils import make_fig, save_fig, COLORS, BRAND_COLORS

R_GAS = 8.314  # J/mol/K


# ── Core calculation ──────────────────────────────────────────────────────────

def compute_resistances(
    T_arr,
    mat_name='SS316L',
    species='HT',
    r_i=0.008,    # m  inner radius
    r_o=0.010,    # m  outer radius
    p_ref=100.0,  # Pa reference HT partial pressure
    P_He=1e5,     # Pa He carrier pressure
    Q_nlpm=5.0,   # Nl/min He flow (single pipe)
):
    """
    Return all four area-specific resistances [m²·s·Pa⁰·⁵/mol] at each T.

    Parameters
    ----------
    T_arr    : array of temperatures [K]
    mat_name : wall material (default 'SS316L')
    species  : hydrogen isotopologue ('HT', 'H2', 'T2')
    r_i      : pipe inner radius [m]
    r_o      : pipe outer radius [m]
    p_ref    : reference partial pressure for √p linearisation [Pa]
    P_He     : He carrier total pressure [Pa]
    Q_nlpm   : He volumetric flow [Nl/min] in a single pipe

    Returns
    -------
    dict of 1-D arrays, each len(T_arr):
        R_wall_flat, R_wall_cyl, R_surface, R_gas_film
    """
    mat       = get_material(mat_name)
    delta     = r_o - r_i
    sqrt_pref = np.sqrt(p_ref)
    d_i       = 2 * r_i

    R_wall_flat = np.full(len(T_arr), np.nan)
    R_wall_cyl  = np.full(len(T_arr), np.nan)
    R_surface   = np.full(len(T_arr), np.nan)
    R_gas_film  = np.full(len(T_arr), np.nan)

    for i, T in enumerate(T_arr):

        # Wall permeability — suppress range warnings (deliberate extrapolation)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Phi = permeability_species(mat, T, species)

        if Phi > 0:
            R_wall_flat[i] = delta / Phi
            # Cylindrical: area-specific to inner surface
            # Derivation: J = Q/(2πr_i L) = Φ/(r_i·ln(r_o/r_i)) · Δ√p
            R_wall_cyl[i]  = r_i * np.log(r_o / r_i) / Phi

        # Surface adsorption: Pd/Ag K_a (Vadrucci 2013), lower bound for SS316L
        # K_a [mol/m²/s/Pa]  →  in √p-space: R = 1/(2·K_a·√p_ref)
        R_a = surface_resistance_H(T)   # = 1/K_a [m²·s·Pa/mol]
        R_surface[i] = R_a / (2.0 * sqrt_pref)

        # Gas-film: He carrier in a single pipe of inner diameter d_i
        # RT/k_m [m²·s·Pa/mol]  →  in √p-space: R = RT/(2·k_m·√p_ref)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mt = mass_transfer_coeff(T, P_He, Q_nlpm, d_i, 1, species)
            R_gas_film[i] = mt['R_gas_film'] / (2.0 * sqrt_pref)
        except Exception:
            pass

    return {
        'R_wall_flat': R_wall_flat,
        'R_wall_cyl':  R_wall_cyl,
        'R_surface':   R_surface,
        'R_gas_film':  R_gas_film,
    }


# ── Plot function ─────────────────────────────────────────────────────────────

def plot_pipe_resistances(
    T_min_K=300, T_max_K=950, n_T=400,
    mat_name='SS316L',
    r_i=0.008, r_o=0.010,
    p_ref=100.0, P_He=1e5, Q_nlpm=5.0,
    theme='light',
    outfile='pipe_resistance_comparison.png',
):
    """
    Plot the four transport resistances vs temperature.

    Parameters
    ----------
    T_min_K, T_max_K : temperature sweep range [K]
    mat_name : wall material key from material_library
    r_i, r_o : inner/outer pipe radius [m]
    p_ref    : reference HT partial pressure for √p linearisation [Pa]
    P_He     : He carrier pressure [Pa]
    Q_nlpm   : He flow in single pipe [Nl/min]
    theme    : 'light' or 'brand'
    outfile  : output filename
    """
    T_arr  = np.linspace(T_min_K, T_max_K, n_T)
    T_C    = T_arr - 273.15

    res    = compute_resistances(
        T_arr, mat_name=mat_name, r_i=r_i, r_o=r_o,
        p_ref=p_ref, P_He=P_He, Q_nlpm=Q_nlpm,
    )

    delta_mm = (r_o - r_i) * 1e3
    OD_mm    = 2 * r_o * 1e3

    title = (
        f'Transport resistance vs temperature — {mat_name} pipe  '
        f'(OD {OD_mm:.0f} mm, wall {delta_mm:.0f} mm)\n'
        f'Resistances in √p-space, linearised at $p_{{\\rm ref}}$ = {p_ref:.0f} Pa  |  '
        f'He: {Q_nlpm:.0f} Nl/min, {P_He/1e5:.1f} bar'
    )

    fig, ax = make_fig(
        xlabel='Temperature (°C)',
        ylabel=r'Area-specific resistance  (m²·s·Pa$^{0.5}$/mol)',
        title=title,
        figsize=(9, 5.5),
        log_y=True,
        theme=theme,
    )

    # Colours
    if theme == 'brand':
        c_flat = BRAND_COLORS['blue_deep']
        c_cyl  = BRAND_COLORS['blue_light']
        c_surf = BRAND_COLORS['orange']
        c_film = BRAND_COLORS['grey']
        c_vline = '#666666'
    else:
        c_flat = COLORS['blue']
        c_cyl  = COLORS['purple']
        c_surf = COLORS['red']
        c_film = COLORS['green']
        c_vline = 'dimgrey'

    ax.plot(T_C, res['R_wall_flat'],
            color=c_flat, lw=2.2,
            label=f'Wall — flat plate  (δ = {delta_mm:.0f} mm)')

    ax.plot(T_C, res['R_wall_cyl'],
            color=c_cyl, lw=2.2, linestyle='--',
            label=f'Wall — cylindrical  (OD {OD_mm:.0f} mm, per unit inner area)')

    ax.plot(T_C, res['R_surface'],
            color=c_surf, lw=1.8,
            label='Surface adsorption  (Pd/Ag $K_a$, Vadrucci 2013 — lower bound)')

    ax.plot(T_C, res['R_gas_film'],
            color=c_film, lw=1.8, linestyle=':',
            label=f'Gas-film  (He, {Q_nlpm:.0f} Nl/min, {P_He/1e5:.0f} bar)')

    # Mark SS316L Arrhenius validity boundary
    T_valid_min_C = get_material(mat_name)['T_min'] - 273.15
    ax.axvline(T_valid_min_C, color=c_vline, lw=1.0, linestyle='--', alpha=0.55)
    ax.text(
        T_valid_min_C - 4, 0.98,
        f'Arrhenius\nvalid →',
        transform=ax.get_xaxis_transform(),
        fontsize=7.5, color=c_vline,
        ha='right', va='top', style='italic',
    )

    ax.legend(fontsize=8.5, loc='upper right')

    save_fig(fig, outfile)
    return fig


SECONDS_PER_DAY = 86400.0   # s/day


# ── Shared contour rendering helpers ─────────────────────────────────────────

def _apply_units(Q_L, units, pipe_length_m=None):
    """
    Convert Q_L [mol/m/s] to the requested plotting units.

    units='mol'  → mol m⁻¹ s⁻¹  (no conversion)
    units='mass' → g T m⁻¹ day⁻¹  (Q_L × 86400 × M_T)
                   if pipe_length_m is given: g T day⁻¹ (total for that run length)

    Returns (Q_plot, clabel_string).
    """
    if units == 'mass':
        Q = Q_L * SECONDS_PER_DAY * M_T          # g T / m / day
        if pipe_length_m is not None:
            Q      = Q * pipe_length_m            # g T / day  (total run)
            clabel = r'Total T loss  (g$_{\rm T}$ day$^{-1}$)'
        else:
            clabel = r'Tritium mass loss  $\dot{m}_T$  (g$_{\rm T}$ m$^{-1}$ day$^{-1}$)'
        return Q, clabel
    return Q_L, r'Permeation rate  $Q_L$  (mol m$^{-1}$ s$^{-1}$)'


def _save_Q_contour(P_2d, Y_2d, Q_plot, xlabel, ylabel, clabel,
                    title, theme, outfile, post_fn=None):
    """
    Build and save a LogNorm contour figure for any Q variant.

    Parameters
    ----------
    P_2d, Y_2d : 2-D meshgrids (x = pressure, y = second variable)
    Q_plot     : 2-D values in the chosen units
    post_fn    : optional callable(ax) for axis annotations
    """
    import matplotlib as mpl
    from plot_utils import make_contour_fig, save_fig

    Q_valid = Q_plot[np.isfinite(Q_plot) & (Q_plot > 0)]
    Q_lo, Q_hi = Q_valid.min(), Q_valid.max()

    fill_levels = np.logspace(np.log10(Q_lo), np.log10(Q_hi), 120)
    exp_lo  = int(np.floor(np.log10(Q_lo)))
    exp_hi  = int(np.ceil(np.log10(Q_hi)))
    c_exps  = list(range(exp_lo, exp_hi + 1))
    c_levels = [10.0 ** e for e in c_exps]
    c_fmt    = {10.0 ** e: f'$10^{{{e}}}$' for e in c_exps}

    fig, ax = make_contour_fig(
        P_2d, Y_2d, Q_plot,
        xlabel=xlabel, ylabel=ylabel, clabel=clabel, title=title,
        figsize=(9, 6),
        log_x=True,
        norm=mpl.colors.LogNorm(vmin=Q_lo, vmax=Q_hi),
        n_fill_levels=fill_levels,
        contour_lines=c_levels,
        contour_fmt=c_fmt,
        contour_fontsize=8,
        cb_ticks=c_levels,
        cb_ticklabels=[f'$10^{{{e}}}$' for e in c_exps],
        theme=theme,
    )

    if post_fn is not None:
        post_fn(ax)

    save_fig(fig, outfile)
    return fig


# ── Pipe flux contour plots ───────────────────────────────────────────────────

def compute_flux_grid(p_arr, od_arr_mm, mat_name, t_mm, T_K, species='HT'):
    """
    Permeation rate per unit pipe length Q_L [mol/m/s] on a (p_in, OD) grid.

    Formula: Q_L = 2π Φ(T) / ln(r_o / r_i) × (√p_in − √p_out),  p_out = 0

    Parameters
    ----------
    p_arr     : 1-D array of inner HT partial pressures [Pa]
    od_arr_mm : 1-D array of pipe outer diameters [mm]
    mat_name  : material key from material_library
    t_mm      : wall thickness [mm]
    T_K       : temperature [K]
    species   : hydrogen isotopologue ('HT', 'H2', 'T2')

    Returns
    -------
    Q_L  : 2-D array, shape (n_OD, n_p) [mol/m/s]
    P_2d : 2-D pressure meshgrid [Pa]
    OD_2d: 2-D OD meshgrid [mm]
    """
    mat = get_material(mat_name)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Phi = permeability_species(mat, T_K, species)

    t_m = t_mm / 1000.0

    # meshgrid: rows = OD index, cols = pressure index
    P_2d, OD_2d = np.meshgrid(p_arr, od_arr_mm)          # shape (n_OD, n_p)
    r_o = OD_2d / 2000.0                                  # m
    r_i = r_o - t_m                                       # m

    # Mask cells where wall thickness >= radius (physically impossible)
    valid    = r_i > 1e-6
    safe_ri  = np.where(valid, r_i, 1e-9)                 # avoid log(0)
    ln_ratio = np.where(valid, np.log(r_o / safe_ri), np.nan)
    Q_L      = np.where(valid, 2.0 * np.pi * Phi / ln_ratio * np.sqrt(P_2d), np.nan)

    return Q_L, P_2d, OD_2d


def plot_pipe_flux_contour(
    mat_name,
    t_mm=2.0,
    T_K=623.0,
    species='HT',
    p_min=0.1,   p_max=1e4,
    od_min_mm=10.0, od_max_mm=100.0,
    n_p=350, n_od=250,
    units='mol',
    pipe_length_m=None,
    loss_limit_g_per_day=None,
    theme='brand',
    outfile=None,
):
    """Contour: Q_L (or mass loss) vs partial pressure and OD at fixed wall and T."""
    p_arr  = np.logspace(np.log10(p_min), np.log10(p_max), n_p)
    od_arr = np.linspace(od_min_mm, od_max_mm, n_od)

    Q_L, P_2d, OD_2d = compute_flux_grid(p_arr, od_arr, mat_name, t_mm, T_K, species)
    Q_plot, clabel    = _apply_units(Q_L, units, pipe_length_m)

    mat_data  = get_material(mat_name)
    T_C       = T_K - 273.15
    T_lo      = mat_data['T_min'] - 273.15
    T_hi      = mat_data['T_max'] - 273.15
    T_warn    = '  ⚠ outside valid range' \
                if not (mat_data['T_min'] <= T_K <= mat_data['T_max']) else ''
    quantity  = 'tritium mass loss' if units == 'mass' else 'permeation rate'
    run_str   = f'total  ({pipe_length_m/1000:.0f} km pipe run)' \
                if pipe_length_m else 'per unit pipe length'

    title = (
        f'{mat_name} — HT {quantity}  {run_str}  '
        f'(cylindrical,  $p_{{\\rm out}} = 0$)\n'
        f'Wall $t$ = {t_mm:.1f} mm  |  '
        f'$T$ = {T_C:.0f} °C  (valid {T_lo:.0f}–{T_hi:.0f} °C){T_warn}'
    )

    if outfile is None:
        mass_tag = '_mass' if units == 'mass' else ''
        len_tag  = f'_{int(pipe_length_m/1000)}km' if pipe_length_m else ''
        outfile  = f'pipe_flux_{mat_name}{mass_tag}{len_tag}.png'

    limit_color = '#FF6666' if theme == 'brand' else '#CC0000'

    def _annotate(ax):
        if loss_limit_g_per_day is not None and units == 'mass':
            cs = ax.contour(P_2d, OD_2d, Q_plot, levels=[loss_limit_g_per_day],
                            colors=[limit_color], linewidths=[2.5], linestyles=['-'])
            try:
                ax.clabel(cs, fmt={loss_limit_g_per_day:
                                   f'  Allowable  {loss_limit_g_per_day:.2e} g/day  '},
                          fontsize=8, inline=True)
            except Exception:
                pass

    return _save_Q_contour(P_2d, OD_2d, Q_plot,
                           'Tritium partial pressure  $p_{\\rm in}$  (Pa)',
                           'Pipe outer diameter  (mm)',
                           clabel, title, theme, outfile, post_fn=_annotate)


# ── Variant: vary wall thickness at fixed OD and temperature ─────────────────

def compute_flux_grid_vs_wall(p_arr, t_arr_mm, mat_name, OD_mm, T_K, species='HT'):
    """
    Q_L [mol/m/s] on a (p_in, wall-thickness) grid at fixed OD and T.

    Returns Q_L shape (n_t, n_p), and matching P_2d, Wall_2d meshgrids.
    """
    mat = get_material(mat_name)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Phi = permeability_species(mat, T_K, species)

    r_o = OD_mm / 2000.0                               # m
    P_2d, Wall_2d = np.meshgrid(p_arr, t_arr_mm)       # shape (n_t, n_p)
    r_i    = r_o - Wall_2d / 1000.0
    valid  = r_i > 1e-6
    safe_ri = np.where(valid, r_i, 1e-9)
    ln_ratio = np.where(valid, np.log(r_o / safe_ri), np.nan)
    Q_L = np.where(valid, 2.0 * np.pi * Phi / ln_ratio * np.sqrt(P_2d), np.nan)
    return Q_L, P_2d, Wall_2d


def plot_pipe_flux_vs_wall(
    mat_name,
    OD_mm=25.4,           # 1 inch
    T_K=623.0,
    species='HT',
    p_min=0.1,  p_max=1e4,
    t_min_mm=0.5, t_max_mm=2.0,
    n_p=350, n_t=200,
    units='mol',
    pipe_length_m=None,
    loss_limit_g_per_day=None,
    theme='brand',
    outfile=None,
):
    """Contour: Q_L (or mass loss) vs (p_in, wall thickness) at fixed OD and T."""
    p_arr = np.logspace(np.log10(p_min), np.log10(p_max), n_p)
    t_arr = np.linspace(t_min_mm, t_max_mm, n_t)

    Q_L, P_2d, Wall_2d = compute_flux_grid_vs_wall(
        p_arr, t_arr, mat_name, OD_mm, T_K, species)
    Q_plot, clabel = _apply_units(Q_L, units, pipe_length_m)

    mat_data  = get_material(mat_name)
    T_C       = T_K - 273.15
    T_lo      = mat_data['T_min'] - 273.15
    T_hi      = mat_data['T_max'] - 273.15
    T_warn    = '  ⚠ outside valid range' \
                if not (mat_data['T_min'] <= T_K <= mat_data['T_max']) else ''
    quantity  = 'tritium mass loss' if units == 'mass' else 'permeation rate'
    run_str   = f'total  ({pipe_length_m/1000:.0f} km pipe run)' \
                if pipe_length_m else 'per unit pipe length'

    title = (
        f'{mat_name} — HT {quantity}  {run_str}  '
        f'(cylindrical,  $p_{{\\rm out}} = 0$)\n'
        f'OD = {OD_mm:.1f} mm  |  '
        f'$T$ = {T_C:.0f} °C  (valid {T_lo:.0f}–{T_hi:.0f} °C){T_warn}'
    )

    if outfile is None:
        mass_tag = '_mass' if units == 'mass' else ''
        len_tag  = f'_{int(pipe_length_m/1000)}km' if pipe_length_m else ''
        outfile  = f'pipe_flux_{mat_name}_vs_wall{mass_tag}{len_tag}.png'

    limit_color = '#FF6666' if theme == 'brand' else '#CC0000'

    def _annotate(ax):
        if loss_limit_g_per_day is not None and units == 'mass':
            cs = ax.contour(P_2d, Wall_2d, Q_plot, levels=[loss_limit_g_per_day],
                            colors=[limit_color], linewidths=[2.5], linestyles=['-'])
            try:
                ax.clabel(cs, fmt={loss_limit_g_per_day:
                                   f'  Allowable  {loss_limit_g_per_day:.2e} g/day  '},
                          fontsize=8, inline=True)
            except Exception:
                pass

    return _save_Q_contour(P_2d, Wall_2d, Q_plot,
                           'Tritium partial pressure  $p_{\\rm in}$  (Pa)',
                           'Wall thickness  (mm)',
                           clabel, title, theme, outfile, post_fn=_annotate)


# ── Variant: vary temperature at fixed OD and wall thickness ──────────────────

def compute_flux_grid_vs_temp(p_arr, T_arr_K, mat_name, OD_mm, t_mm, species='HT'):
    """
    Q_L [mol/m/s] on a (p_in, T) grid at fixed OD and wall thickness.

    Returns Q_L shape (n_T, n_p), and matching P_2d [Pa], Temp_2d [°C] meshgrids.
    """
    mat  = get_material(mat_name)
    r_o  = OD_mm / 2000.0
    r_i  = r_o - t_mm / 1000.0
    if r_i <= 0:
        raise ValueError(f"Wall t={t_mm} mm >= radius for OD={OD_mm} mm")
    ln_ratio = np.log(r_o / r_i)

    Q_L = np.zeros((len(T_arr_K), len(p_arr)))
    for i, T in enumerate(T_arr_K):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Phi = permeability_species(mat, T, species)
        Q_L[i, :] = 2.0 * np.pi * Phi / ln_ratio * np.sqrt(p_arr)

    P_2d, Temp_2d = np.meshgrid(p_arr, T_arr_K - 273.15)   # Temp in °C
    return Q_L, P_2d, Temp_2d


def plot_pipe_flux_vs_temperature(
    mat_name,
    OD_mm=25.4,           # 1 inch
    t_mm=2.0,
    species='HT',
    p_min=0.1,  p_max=1e4,
    T_min_C=100.0, T_max_C=700.0,
    n_p=350, n_T=200,
    units='mol',
    pipe_length_m=None,
    loss_limit_g_per_day=None,
    theme='brand',
    outfile=None,
):
    """Contour: Q_L (or mass loss) vs (p_in, temperature) at fixed OD and wall."""
    p_arr   = np.logspace(np.log10(p_min), np.log10(p_max), n_p)
    T_arr_K = np.linspace(T_min_C + 273.15, T_max_C + 273.15, n_T)

    Q_L, P_2d, Temp_2d = compute_flux_grid_vs_temp(
        p_arr, T_arr_K, mat_name, OD_mm, t_mm, species)
    Q_plot, clabel = _apply_units(Q_L, units, pipe_length_m)

    mat_data   = get_material(mat_name)
    T_valid_lo = mat_data['T_min'] - 273.15
    T_valid_hi = mat_data['T_max'] - 273.15
    quantity   = 'tritium mass loss' if units == 'mass' else 'permeation rate'
    run_str    = f'total  ({pipe_length_m/1000:.0f} km pipe run)' \
                 if pipe_length_m else 'per unit pipe length'

    title = (
        f'{mat_name} — HT {quantity}  {run_str}  '
        f'(cylindrical,  $p_{{\\rm out}} = 0$)\n'
        f'OD = {OD_mm:.1f} mm  |  '
        f'Wall $t$ = {t_mm:.1f} mm  |  '
        f'Arrhenius valid {T_valid_lo:.0f}–{T_valid_hi:.0f} °C'
    )

    if outfile is None:
        mass_tag = '_mass' if units == 'mass' else ''
        len_tag  = f'_{int(pipe_length_m/1000)}km' if pipe_length_m else ''
        outfile  = f'pipe_flux_{mat_name}_vs_temp{mass_tag}{len_tag}.png'

    text_color  = 'white' if theme == 'brand' else '#444444'
    limit_color = '#FF6666' if theme == 'brand' else '#CC0000'
    span = T_max_C - T_min_C

    def _annotate(ax):
        if T_min_C < T_valid_lo < T_max_C:
            ax.axhline(T_valid_lo, color=text_color, lw=1.0, ls='--', alpha=0.6)
            ax.text(p_min * 3, T_valid_lo + span * 0.02,
                    f'Arrhenius valid above {T_valid_lo:.0f} °C',
                    color=text_color, fontsize=7.5, va='bottom')
        if mat_name == 'RAFM' and T_min_C < 300.0 < T_max_C:
            ax.axhline(300.0, color=text_color, lw=0.8, ls=':', alpha=0.55)
            ax.text(p_min * 3, 300.0 + span * 0.02,
                    'Trapping significant below 300 °C',
                    color=text_color, fontsize=7.5, va='bottom')
        if loss_limit_g_per_day is not None and units == 'mass':
            cs = ax.contour(P_2d, Temp_2d, Q_plot, levels=[loss_limit_g_per_day],
                            colors=[limit_color], linewidths=[2.5], linestyles=['-'])
            try:
                ax.clabel(cs, fmt={loss_limit_g_per_day:
                                   f'  Allowable  {loss_limit_g_per_day:.2e} g/day  '},
                          fontsize=8, inline=True)
            except Exception:
                pass

    return _save_Q_contour(P_2d, Temp_2d, Q_plot,
                           'Tritium partial pressure  $p_{\\rm in}$  (Pa)',
                           'Temperature  (°C)',
                           clabel, title, theme, outfile, post_fn=_annotate)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os

    # ── Output directories ────────────────────────────────────────────────────
    ROOT = 'outputs'
    for d in [
        f'{ROOT}/resistance',
        f'{ROOT}/SS316L/mol_rate',
        f'{ROOT}/SS316L/mass_per_m',
        f'{ROOT}/SS316L/10km_with_limit',
        f'{ROOT}/RAFM/mol_rate',
        f'{ROOT}/RAFM/mass_per_m',
        f'{ROOT}/RAFM/10km_with_limit',
    ]:
        os.makedirs(d, exist_ok=True)

    # ── Resistance comparison ─────────────────────────────────────────────────
    plot_pipe_resistances(theme='light',
                          outfile=f'{ROOT}/resistance/pipe_resistance_comparison.png')
    plot_pipe_resistances(theme='brand',
                          outfile=f'{ROOT}/resistance/pipe_resistance_comparison_brand.png')

    # ── Permeation rate [mol/m/s] — light + brand ─────────────────────────────
    for mat in ('SS316L', 'RAFM'):
        D = f'{ROOT}/{mat}/mol_rate'
        plot_pipe_flux_contour(mat, t_mm=2.0, T_K=623.0,
                               theme='light', outfile=f'{D}/p_vs_OD_light.png')
        plot_pipe_flux_contour(mat, t_mm=2.0, T_K=623.0,
                               theme='brand', outfile=f'{D}/p_vs_OD_brand.png')
        plot_pipe_flux_vs_wall(mat, OD_mm=25.4, T_K=623.0,
                               t_min_mm=0.5, t_max_mm=2.0,
                               theme='light', outfile=f'{D}/p_vs_wall_light.png')
        plot_pipe_flux_vs_wall(mat, OD_mm=25.4, T_K=623.0,
                               t_min_mm=0.5, t_max_mm=2.0,
                               theme='brand', outfile=f'{D}/p_vs_wall_brand.png')
        plot_pipe_flux_vs_temperature(mat, OD_mm=25.4, t_mm=2.0,
                                      T_min_C=100.0, T_max_C=700.0,
                                      theme='light', outfile=f'{D}/p_vs_temp_light.png')
        plot_pipe_flux_vs_temperature(mat, OD_mm=25.4, t_mm=2.0,
                                      T_min_C=100.0, T_max_C=700.0,
                                      theme='brand', outfile=f'{D}/p_vs_temp_brand.png')

    # ── Mass loss [g T/m/day] — brand only ───────────────────────────────────
    for mat in ('SS316L', 'RAFM'):
        D = f'{ROOT}/{mat}/mass_per_m'
        plot_pipe_flux_contour(mat, t_mm=2.0, T_K=623.0,
                               units='mass', theme='brand',
                               outfile=f'{D}/p_vs_OD_brand.png')
        plot_pipe_flux_vs_wall(mat, OD_mm=25.4, T_K=623.0,
                               t_min_mm=0.5, t_max_mm=2.0,
                               units='mass', theme='brand',
                               outfile=f'{D}/p_vs_wall_brand.png')
        plot_pipe_flux_vs_temperature(mat, OD_mm=25.4, t_mm=2.0,
                                      T_min_C=100.0, T_max_C=700.0,
                                      units='mass', theme='brand',
                                      outfile=f'{D}/p_vs_temp_brand.png')

    # ── Total loss — 10 km run with allowable limit ───────────────────────────
    L_10KM = 10_000.0   # m
    LIMIT  = 0.77       # g T / day
    for mat in ('SS316L', 'RAFM'):
        D = f'{ROOT}/{mat}/10km_with_limit'
        plot_pipe_flux_contour(mat, t_mm=2.0, T_K=623.0,
                               units='mass', pipe_length_m=L_10KM,
                               loss_limit_g_per_day=LIMIT, theme='brand',
                               outfile=f'{D}/p_vs_OD.png')
        plot_pipe_flux_vs_wall(mat, OD_mm=25.4, T_K=623.0,
                               t_min_mm=0.5, t_max_mm=2.0,
                               units='mass', pipe_length_m=L_10KM,
                               loss_limit_g_per_day=LIMIT, theme='brand',
                               outfile=f'{D}/p_vs_wall.png')
        plot_pipe_flux_vs_temperature(mat, OD_mm=25.4, t_mm=2.0,
                                      T_min_C=100.0, T_max_C=700.0,
                                      units='mass', pipe_length_m=L_10KM,
                                      loss_limit_g_per_day=LIMIT, theme='brand',
                                      outfile=f'{D}/p_vs_temp.png')

    print('Done.')
