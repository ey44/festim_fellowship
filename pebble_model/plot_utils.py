# plot_utils.py
# =============================================================================
# Plotting conventions.
#
# RULE: one plot per figure, always.
# Every plot in this package is created through make_fig() or make_contour_fig(),
# which return a (fig, ax) pair for a single-axis figure.
#
# Two themes are available:
#
#   theme='light'  (default) — white background, good for reports/papers
#   theme='brand'            — Type One Energy dark background (#26252C),
#                              blue→orange colourmap, good for presentations
#
# Usage — standard line/scatter plot:
#
#   from tes_sim.plot_utils import make_fig, save_fig, COLORS, BRAND_COLORS
#
#   fig, ax = make_fig(
#       xlabel="Purge He flow rate (kg/s)",
#       ylabel="Tritium partial pressure (Pa)",
#       title="P_T vs purge flow — Stage 1",
#       theme='brand',          # optional, default is 'light'
#   )
#   ax.plot(x, y, 'o-', color=BRAND_COLORS['orange'])
#   save_fig(fig, "stage1_PT_vs_mdot.png")
#
# Usage — contour plot:
#
#   from plot_utils import make_contour_fig, save_fig
#
#   fig, ax = make_contour_fig(
#       X, Y, Z,
#       xlabel="TES inlet temperature (°C)",
#       ylabel="Area (m²)",
#       clabel="Hot-side outlet temperature (°C)",
#       contour_lines=[100, 200, 300, 400, 500],
#       contour_fmt="{:.0f} °C",
#       log_y=True,
#       theme='brand',
#   )
#   save_fig(fig, "economiser.png")
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------------------------------------------------------
# Global style defaults (light theme base — overridden per-call for brand theme)
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family'      : 'sans-serif',
    'font.size'        : 11,
    'axes.titlesize'   : 11,
    'axes.labelsize'   : 11,
    'xtick.labelsize'  : 10,
    'ytick.labelsize'  : 10,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
    'figure.dpi'       : 100,
    'savefig.dpi'      : 150,
    'savefig.bbox'     : 'tight',
    'lines.linewidth'  : 1.5,
    'lines.markersize' : 4,
})

# -----------------------------------------------------------------------------
# Colour palettes
# -----------------------------------------------------------------------------

# Original neutral palette — unchanged for backward compatibility
COLORS = {
    'blue'   : 'steelblue',
    'red'    : 'firebrick',
    'orange' : 'darkorange',
    'green'  : 'seagreen',
    'purple' : 'mediumpurple',
    'grey'   : 'dimgrey',
}

# Type One Energy brand palette
BRAND_COLORS = {
    'dark'       : '#26252C',   # background
    'blue_deep'  : '#3881C3',   # primary blue
    'blue_mid'   : '#48A0D8',   # secondary blue
    'blue_light' : '#77CCF3',   # accent blue
    'orange'     : '#FC9D06',   # highlight / warm end of colourmap
    'white'      : '#FFFFFF',
    'grey'       : '#888888',
}

# Brand line colour cycle — used automatically in brand theme
BRAND_CYCLE = [
    BRAND_COLORS['blue_deep'],
    BRAND_COLORS['orange'],
    BRAND_COLORS['blue_light'],
    BRAND_COLORS['blue_mid'],
    BRAND_COLORS['white'],
    BRAND_COLORS['grey'],
]

# Brand colourmap: deep blue → light blue → pale gold → orange
# Four stops give a much more distinct gradient across the mid-range;
# the near-white pale gold (~#F0DFA0) acts as a perceptual midpoint so
# moderate values are never ambiguous between the blue and orange ends.
BRAND_CMAP = LinearSegmentedColormap.from_list(
    't1e',
    [
        BRAND_COLORS['blue_deep'],   # low     : #3881C3
        BRAND_COLORS['blue_light'],  # mid-low : #77CCF3
        '#F0DFA0',                   # mid-high: pale gold (perceptual centre)
        BRAND_COLORS['orange'],      # high    : #FC9D06
    ],
    N=512,
)

# Light-theme colourmap (viridis-like, works on white background)
LIGHT_CMAP = 'viridis'

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _apply_brand_theme(fig, ax):
    """Apply Type One Energy dark theme to an existing fig/ax pair."""
    fig.patch.set_facecolor(BRAND_COLORS['dark'])
    ax.set_facecolor(BRAND_COLORS['dark'])
    ax.tick_params(colors=BRAND_COLORS['white'], labelsize=10, which='both')
    ax.xaxis.label.set_color(BRAND_COLORS['white'])
    ax.yaxis.label.set_color(BRAND_COLORS['white'])
    ax.title.set_color(BRAND_COLORS['white'])
    for sp in ax.spines.values():
        sp.set_edgecolor('#555555')
    ax.set_prop_cycle(color=BRAND_CYCLE)
    # Lighten grid for dark background
    ax.grid(True, alpha=0.2, linestyle='--', color='#888888')


def _apply_light_theme(fig, ax):
    """Ensure clean light theme (resets any lingering state)."""
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.tick_params(colors='black', labelsize=10, which='both')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.title.set_color('black')
    for sp in ax.spines.values():
        sp.set_edgecolor('#cccccc')
    ax.set_prop_cycle(color=list(COLORS.values()))
    ax.grid(True, alpha=0.3, linestyle='--', color='#cccccc')


def _style_colorbar(cb, theme):
    """Style a colorbar to match the chosen theme."""
    text_color = BRAND_COLORS['white'] if theme == 'brand' else 'black'
    cb.ax.yaxis.set_tick_params(color=text_color)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=text_color, fontsize=9)
    cb.set_label(cb.ax.get_ylabel() or '', color=text_color, fontsize=10)


# -----------------------------------------------------------------------------
# Public API — standard plots
# -----------------------------------------------------------------------------

def make_fig(xlabel='', ylabel='', title='', figsize=(6, 4), theme='light',
             log_x=False, log_y=False):
    """
    Create a single-axis figure — the only way to make a standard plot in tes_sim.

    Parameters
    ----------
    xlabel  : str    x-axis label
    ylabel  : str    y-axis label
    title   : str    axis title (use LaTeX via $...$ for symbols)
    figsize : tuple  (width, height) in inches. Default (6, 4).
    theme   : str    'light' (default, white bg) or 'brand' (T1E dark bg)
    log_x   : bool   use log scale on x-axis
    log_y   : bool   use log scale on y-axis

    Returns
    -------
    fig : matplotlib Figure
    ax  : matplotlib Axes

    Example
    -------
    fig, ax = make_fig(
        xlabel="Temperature (°C)",
        ylabel="Permeation flux (mol/s)",
        title="First-wall permeation vs temperature",
        theme='brand',
    )
    ax.plot(T, J, 'o-', color=BRAND_COLORS['orange'])
    save_fig(fig, "permeation_vs_T.png")
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if theme == 'brand':
        _apply_brand_theme(fig, ax)
    else:
        _apply_light_theme(fig, ax)

    if log_x:
        ax.set_xscale('log')
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    if log_y:
        ax.set_yscale('log')
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    return fig, ax


# -----------------------------------------------------------------------------
# Public API — contour plots
# -----------------------------------------------------------------------------

def make_contour_fig(
    X, Y, Z,
    xlabel='', ylabel='', title='', clabel='',
    figsize=(9, 7),
    theme='brand',
    log_y=False,
    log_x=False,
    cmap=None,
    vmin=None, vmax=None,
    norm=None,
    n_fill_levels=100,
    contour_lines=None,
    contour_fmt=None,
    contour_lw=1.2,
    contour_color='white',
    contour_fontsize=9,
    cb_ticks=None,
    cb_ticklabels=None,
):
    """
    Create a single filled-contour figure with optional contour line overlays.

    Parameters
    ----------
    X, Y, Z         : 2-D arrays  meshgrid inputs and values to colour
    xlabel          : str         x-axis label
    ylabel          : str         y-axis label
    title           : str         axis title
    clabel          : str         colourbar label
    figsize         : tuple       figure size in inches
    theme           : str         'light' or 'brand'
    log_y           : bool        log scale on y-axis
    log_x           : bool        log scale on x-axis
    cmap            : colormap    override default (brand→BRAND_CMAP, light→viridis)
    vmin, vmax      : float       colourbar limits
    norm            : Normalize   matplotlib normalisation, e.g.
                                  ``mpl.colors.LogNorm()`` or
                                  ``mpl.colors.PowerNorm(gamma=0.5)``
                                  for skewed data distributions.
                                  Overrides vmin/vmax if supplied.
    n_fill_levels   : int         number of filled contour levels (default 100)
    contour_lines   : list        Z values at which to draw labelled contour lines
    contour_fmt     : str or dict label format string e.g. "{:.0f} °C"
                                  or dict {value: label_string}
    contour_lw      : float       contour line width
    contour_color   : str         contour line colour (default 'white')
    contour_fontsize: int         contour label font size
    cb_ticks        : list        explicit colourbar tick positions
    cb_ticklabels   : list        explicit colourbar tick labels

    Returns
    -------
    fig : matplotlib Figure
    ax  : matplotlib Axes

    Example
    -------
    fig, ax = make_contour_fig(
        T_TES_2d, A_2d, T_ho_2d,
        xlabel="TES inlet temperature (°C)",
        ylabel="Area (m², log scale)",
        clabel="Hot-side outlet temperature (°C)",
        contour_lines=[100, 200, 300, 400, 500],
        contour_fmt="{:.0f} °C",
        log_y=True,
        theme='brand',
    )
    save_fig(fig, "economiser.png")
    """
    # Choose colourmap
    if cmap is None:
        cmap = BRAND_CMAP if theme == 'brand' else LIGHT_CMAP

    # Choose contour line colour sensibly for light theme
    if theme == 'light' and contour_color == 'white':
        contour_color = '#333333'

    fig, ax = plt.subplots(figsize=figsize)

    if theme == 'brand':
        _apply_brand_theme(fig, ax)
    else:
        _apply_light_theme(fig, ax)

    # Log axes
    if log_y:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    if log_x:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    # Filled contour
    cf = ax.contourf(X, Y, Z,
                     levels=n_fill_levels,
                     cmap=cmap,
                     norm=norm,
                     vmin=vmin, vmax=vmax,
                     alpha=0.93)

    # Contour lines + labels
    if contour_lines is not None:
        cs = ax.contour(X, Y, Z,
                        levels=contour_lines,
                        colors=contour_color,
                        linewidths=contour_lw,
                        linestyles='--',
                        alpha=0.85)

        # Build format dict
        if contour_fmt is None:
            fmt_dict = {v: f'{v}' for v in contour_lines}
        elif isinstance(contour_fmt, str):
            fmt_dict = {v: contour_fmt.format(v) for v in contour_lines}
        else:
            fmt_dict = contour_fmt  # user supplied dict directly

        ax.clabel(cs,
                  fmt=fmt_dict,
                  fontsize=contour_fontsize,
                  inline=True,
                  colors=contour_color)

    # Colourbar
    cb = fig.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label(clabel,
                 color=BRAND_COLORS['white'] if theme == 'brand' else 'black',
                 fontsize=10)
    _style_colorbar(cb, theme)
    if cb_ticks is not None:
        cb.set_ticks(cb_ticks)
    if cb_ticklabels is not None:
        cb.set_ticklabels(cb_ticklabels)

    # Axis labels / title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Re-apply label colours after contourf (matplotlib can reset them)
    text_color = BRAND_COLORS['white'] if theme == 'brand' else 'black'
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)
    ax.title.set_color(text_color)

    return fig, ax


# -----------------------------------------------------------------------------
# Public API — stacked bar charts
# -----------------------------------------------------------------------------

def make_bar_chart(
    data,
    bar_labels,
    layer_colors,
    layer_labels=None,
    bar_width=0.52,
    xlabel='',
    ylabel='',
    title='',
    figsize=(10, 6),
    theme='light',
    reference_line=None,
    reference_label=None,
    annotate_totals=True,
    totals_denom=None,
    legend_title='',
    legend_kwargs=None,
):
    """
    Create a stacked bar chart — one bar per entry in bar_labels, one stack
    layer per row in data.

    Parameters
    ----------
    data           : array-like, shape (n_layers, n_bars)
                     Each row is one stack layer (e.g. one criterion).
                     Values are the bar heights for that layer.
    bar_labels     : list of str
                     x-axis tick labels, one per bar (n_bars).
    layer_colors   : list of str
                     Colours for each stack layer, length n_layers.
    layer_labels   : list of str, optional
                     Legend labels for each layer. If None, no legend is drawn.
    bar_width      : float    Width of each bar (default 0.52).
    xlabel         : str      x-axis label.
    ylabel         : str      y-axis label.
    title          : str      Axis title (empty string to suppress).
    figsize        : tuple    Figure size in inches.
    theme          : str      'light' (default) or 'brand'.
    reference_line : float    If set, draw a horizontal dashed reference line.
    reference_label: str      Label placed next to the reference line.
    annotate_totals: bool     If True, write the column total above each bar.
    totals_denom   : int/float
                     If set, totals are formatted as "total / totals_denom".
    legend_title   : str      Title string for the legend box.
    legend_kwargs  : dict     Extra keyword arguments forwarded to ax.legend().

    Returns
    -------
    fig : matplotlib Figure
    ax  : matplotlib Axes

    Example
    -------
    fig, ax = make_bar_chart(
        data=weighted_scores,          # shape (n_criteria, n_technologies)
        bar_labels=technology_names,
        layer_colors=CRIT_COLORS,
        layer_labels=criteria_names,
        ylabel="Weighted Score",
        theme='light',
        reference_line=155,
        reference_label="Max possible: 155",
        annotate_totals=True,
        totals_denom=155,
        legend_title="Criterion  (w = weighting)",
    )
    save_fig(fig, "scoring_chart.png", folder="outputs")
    """
    import matplotlib.patches as mpatches

    data = np.asarray(data, dtype=float)
    n_layers, n_bars = data.shape

    fig, ax = make_fig(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        figsize=figsize,
        theme=theme,
    )

    text_color = BRAND_COLORS['white'] if theme == 'brand' else '#222222'
    ref_color  = BRAND_COLORS['grey']  if theme == 'brand' else COLORS['grey']

    x = np.arange(n_bars)
    bottoms = np.zeros(n_bars)

    for i in range(n_layers):
        ax.bar(
            x, data[i], bar_width,
            bottom=bottoms,
            color=layer_colors[i],
            edgecolor=BRAND_COLORS['dark'] if theme == 'brand' else 'white',
            linewidth=0.5,
            zorder=3,
        )
        bottoms += data[i]

    totals = data.sum(axis=0)

    # Reference line
    if reference_line is not None:
        ax.axhline(reference_line, color=ref_color, linewidth=1.0,
                   linestyle='--', zorder=2, alpha=0.7)
        if reference_label:
            ax.text(
                n_bars - 0.5, reference_line + (reference_line * 0.008),
                reference_label,
                fontsize=8.5, color=ref_color,
                va='bottom', ha='right', fontstyle='italic',
            )

    # Total annotations
    if annotate_totals:
        for j, total in enumerate(totals):
            label = (f"{int(total)} / {totals_denom}"
                     if totals_denom is not None else f"{int(total)}")
            y_offset = reference_line * 0.01 if reference_line else totals.max() * 0.01
            ax.text(
                x[j], total + y_offset,
                label,
                ha='center', va='bottom',
                fontsize=10, fontweight='bold',
                color=text_color,
            )

    # Axes formatting
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=10)
    ax.set_xlim(-0.5, n_bars - 0.5)
    y_top = (reference_line if reference_line else totals.max()) * 1.13
    ax.set_ylim(0, y_top)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Legend
    if layer_labels is not None:
        patches = [
            mpatches.Patch(color=layer_colors[i], label=layer_labels[i])
            for i in range(n_layers)
        ]
        kw = dict(
            loc='upper left',
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            framealpha=0.9,
            edgecolor='#cccccc',
            fontsize=8.5,
            title=legend_title,
            title_fontsize=9,
        )
        if legend_kwargs:
            kw.update(legend_kwargs)
        ax.legend(handles=patches, **kw)

    return fig, ax


# -----------------------------------------------------------------------------
# Public API — save
# -----------------------------------------------------------------------------

def save_fig(fig, filename, folder='.'):
    """
    Save a figure and print confirmation.

    Parameters
    ----------
    fig      : matplotlib Figure
    filename : str   e.g. "stage1_PT_vs_mdot.png"
    folder   : str   directory to save into. Default: current directory.
    """
    path = os.path.join(folder, filename)
    fig.tight_layout()
    fig.savefig(path)
    print(f'  Saved: {path}')
    plt.close(fig)
