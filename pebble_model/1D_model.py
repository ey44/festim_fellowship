'''
Initial model to work with the system in 1D.

The first will be a coated pebble with a coating of 0.1 mm and a pebble radius of 0.5 mm.

There are clear uncertainties on the material properties.

The model is a fixed temperature, with a tritium source term in the bulk, and a surface reaction boundary condition on the coating surface.

No traps added yet.
Need to consider porosity too.

'''


import os

import festim as F
import numpy as np
import festim as F
from material_library import Zr, Li2O

script_dir = os.path.dirname(os.path.abspath(__file__))

my_model = F.HydrogenTransportProblemDiscontinuous()

# Define model inputs
coating_thickness = 1e-4
pebble_radius = 5e-4
local_refinement_thickness = 5e-7
surface_location = pebble_radius + coating_thickness
model_temperature = 573.0 # blanket inlet temperature
p_total = 2e5 # purge gas pressure [Pa]
x_D2_ppm = 100 
x_D2 = x_D2_ppm * 1e-6 # convert to fraction
p_D2 = x_D2 * p_total
p_DT = 0.0
p_T2 = 0.0
surface_recombination = True

# Select materials



l2O_festim = F.Material(D_0=Li2O.D_0, E_D=Li2O.E_D,
                 K_S_0=1e-3 * 6.022e23, E_K_S=Zr.E_K_S, # placeholder values for solubility, as the Li2O solubility is not well known
                 solubility_law="sievert")
Zr_festim = F.Material(D_0=Zr.D_0, E_D=Zr.E_D,
                     K_S_0=Zr.K_S_0, E_K_S=Zr.E_K_S,
                     solubility_law="sievert")


# print the solubility, and diffusivity values at the simulation temperature
for name, mat in [("Li2O (ceramic)", l2O_festim), ("Zr (coating)", Zr_festim)]:
    D = mat.D_0 * np.exp(-mat.E_D / (F.k_B * model_temperature))
    K_S = mat.K_S_0 * np.exp(-mat.E_K_S / (F.k_B * model_temperature))
    print(f"{name}: D_0 = {mat.D_0:.3e} m^2/s, E_D = {mat.E_D:.3e} eV, "
          f"K_S_0 = {mat.K_S_0:.3e} atoms/m^3/Pa^0.5, E_K_S = {mat.E_K_S:.3e} eV")
    print(f"{name}: D({model_temperature:.1f} K) = {D:.3e} m^2/s, "
          f"K_S({model_temperature:.1f} K) = {K_S:.3e} atoms/m^3/Pa^0.5")


# Define mesh
vertices = np.unique(
    np.concatenate(
        [
            np.linspace(
                0.0,
                pebble_radius ,
                num=150,
            ),
            np.linspace(
                pebble_radius,
                surface_location,
                num=100,
            ),

        ]
    )
)

mesh = F.Mesh1D(
    vertices=vertices,
    coordinate_system="spherical",
)

my_model.mesh = mesh

# Define volume and surface subdomains
ceramic_subdomain = F.VolumeSubdomain1D(
    id=1,
    borders=[0.0, pebble_radius],
    material=l2O_festim,
)

coating_subdomain = F.VolumeSubdomain1D(
    id=2,
    borders=[pebble_radius, surface_location],
    material=Zr_festim,
)

surface = F.SurfaceSubdomain1D(
    id=3,
    x=surface_location,
)

my_model.subdomains = [
    ceramic_subdomain,
    coating_subdomain,
    surface,
]

# Species exist in both materials
T = F.Species(
    name="T",
    subdomains=[ceramic_subdomain, coating_subdomain],
)

D = F.Species(
    name="D",
    subdomains=[ceramic_subdomain, coating_subdomain],
)

my_model.species = [T, D]

# Couple the separate ceramic and coating solutions
core_coating_interface = F.Interface(
    id=4,
    subdomains=[ceramic_subdomain, coating_subdomain],
    penalty_term=1e20,
    method="penalty",
)

my_model.interfaces = [core_coating_interface]


# Define boundary conditions at the surface for release
# Recombination/dissociation coefficients: K_r = k_r0 * exp(-E_kr / k_B / T),
# K_d = k_d0 * exp(-E_kd / k_B / T). Isotope effects (D vs T) are neglected here,
# so the same coefficients are reused for all three reactions.
# TODO: replace with real recombination data for T/D on Zr.
# Atomic concentration convention: c in atoms/m3, flux in atoms/m2/s

if surface_recombination:

    k_r0_common = 1e-22   # 
    E_kr_common = 0.0     # eV
    k_d0_common = 1e-22  # pre-exponential dissociation coefficient (m^-2/s/Pa)
    E_kd_common = E_kr_common  # dissociation activation energy (eV)

    k_r0_D2 = k_r0_DT = k_r0_T2 = k_r0_common
    E_kr_D2 = E_kr_DT = E_kr_T2 = E_kr_common
    k_d0_D2 = k_d0_DT = k_d0_T2 = k_d0_common
    E_kd_D2 = E_kd_DT = E_kd_T2 = E_kd_common

    D2_bc = F.SurfaceReactionBC(
        reactant=[D, D],
        gas_pressure=p_D2,
        k_r0=k_r0_D2,
        E_kr=E_kr_D2,
        k_d0=k_d0_D2,
        E_kd=E_kd_D2,
        subdomain=surface,
    )

    DT_bc = F.SurfaceReactionBC(
        reactant=[D, T],
        gas_pressure=p_DT,
        k_r0=k_r0_DT,
        E_kr=E_kr_DT,
        k_d0=k_d0_DT,
        E_kd=E_kd_DT,
        subdomain=surface,
    )

    T2_bc = F.SurfaceReactionBC(
        reactant=[T, T],
        gas_pressure=p_T2,
        k_r0=k_r0_T2,
        E_kr=E_kr_T2,
        k_d0=k_d0_T2,
        E_kd=E_kd_T2,
        subdomain=surface,
    )

    my_model.boundary_conditions = [D2_bc, DT_bc, T2_bc]

else:
    D_sieverts_bc = F.SievertsBC(
        subdomain=surface,
        S_0=Zr_festim.K_S_0,
        E_S=Zr_festim.E_K_S,
        pressure=p_D2,
        species=D,
    )

    T_sieverts_bc = F.SievertsBC(
        subdomain=surface,
        S_0=Zr_festim.K_S_0,
        E_S=Zr_festim.E_K_S,
        pressure=p_T2,
        species=T,
    )

    my_model.boundary_conditions = [
        D_sieverts_bc,
        T_sieverts_bc,
    ]

# Define temperature

my_model.temperature = model_temperature # inlet blanket conditions

# Defint source
# volumetric tritium generation rate S [particles/m3/s]
P_fus = 1140 # MW
TBR = 1.2
blanket_volume = 1500 # m3
true_pebbles_volume_fraction = 0.5
pebbles_volume = blanket_volume * true_pebbles_volume_fraction
tritium_production_rate = P_fus * TBR * 6.2415 * 10**24 / 17.6e6  # tritium atoms/s

volume_per_pebble = (4 / 3) * np.pi * pebble_radius**3
number_of_pebbles = pebbles_volume / volume_per_pebble

# volume_per_pebble cancels out here: S is a density (atoms/m3/s) applied uniformly
# across the ceramic subdomain, so it's just total production over total breeder volume,
# independent of how many pebbles that volume is divided into.
S = tritium_production_rate / pebbles_volume

my_model.sources = [
    F.ParticleSource(value=S, volume=ceramic_subdomain, species=T),
]

# Define exports so the concentration profiles can be plotted after solving
T_profile_ceramic = F.Profile1DExport(field=T, subdomain=ceramic_subdomain)
D_profile_ceramic = F.Profile1DExport(field=D, subdomain=ceramic_subdomain)
T_profile_coating = F.Profile1DExport(field=T, subdomain=coating_subdomain)
D_profile_coating = F.Profile1DExport(field=D, subdomain=coating_subdomain)


ceramic_vtx = F.VTXSpeciesExport(filename=os.path.join(script_dir, "ceramic.bp"), field=[T, D], subdomain=ceramic_subdomain)
coating_vtx = F.VTXSpeciesExport(filename=os.path.join(script_dir, "coating.bp"), field=[T, D], subdomain=coating_subdomain)

my_model.exports = [T_profile_ceramic,
                    D_profile_ceramic,
                    T_profile_coating,
                    D_profile_coating,
                    ceramic_vtx,
                    coating_vtx,
]

surface_flux = F.SurfaceFlux(field=T, surface=surface)
my_model.exports.append(surface_flux)




# Define model settings
my_model.settings = F.Settings(
    atol=1e10,
    rtol=1e-8,
    max_iterations=500,
    transient=True,
    final_time=1e5,
    stepsize=F.Stepsize(
        initial_value=1e-1,
        growth_factor=1.5,
        cutback_factor=0.5,
        target_nb_iterations=5,
        max_stepsize=1e3,
    ),
)
my_model.initialise()
my_model.run()

# Plot the concentration profiles at the final timestep
from plot_utils import make_fig, save_fig, COLORS, LIGHT_CMAP
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Plot the ceramic and coating subdomain profiles directly on the same axes
# (as two separate line segments) rather than stitching them into one array.
# This keeps the interface jump clean instead of relying on argsort tie-breaking
# between two subdomains that share the same x value at the interface.
t = T_profile_ceramic.t

fig, ax = make_fig(
    xlabel="x (m)",
    ylabel="T concentration (m$^{-3}$)",
    title=f"T concentration profile (t = {t[-1]:.3g} s)",
    log_y=True,
)
ax.plot(T_profile_ceramic.x, T_profile_ceramic.data[-1], color=COLORS['red'])
ax.plot(T_profile_coating.x, T_profile_coating.data[-1], color=COLORS['red'])
save_fig(fig, "T_concentration.png")

fig, ax = make_fig(
    xlabel="x (m)",
    ylabel="D concentration (m$^{-3}$)",
    title=f"D concentration profile (t = {t[-1]:.3g} s)",
    log_y=True,
)
ax.plot(D_profile_ceramic.x, D_profile_ceramic.data[-1], color=COLORS['blue'])
ax.plot(D_profile_coating.x, D_profile_coating.data[-1], color=COLORS['blue'])
save_fig(fig, "D_concentration.png")

# Zoom in on the ceramic/coating interface (+/-2% of the total radial extent)
# to inspect the concentration jump across it, one plot per species
interface_half_width = 0.0002 * surface_location

fig, ax = make_fig(
    xlabel="x (m)",
    ylabel="T concentration (m$^{-3}$)",
    title=f"T concentration near interface (t = {t[-1]:.3g} s)",
    log_y=True,
)
ax.plot(T_profile_ceramic.x, T_profile_ceramic.data[-1], color=COLORS['red'])
ax.plot(T_profile_coating.x, T_profile_coating.data[-1], color=COLORS['red'])
ax.set_xlim(pebble_radius - interface_half_width, pebble_radius + interface_half_width)
save_fig(fig, "T_interface_concentration.png")

fig, ax = make_fig(
    xlabel="x (m)",
    ylabel="D concentration (m$^{-3}$)",
    title=f"D concentration near interface (t = {t[-1]:.3g} s)",
    log_y=True,
)
ax.plot(D_profile_ceramic.x, D_profile_ceramic.data[-1], color=COLORS['blue'])
ax.plot(D_profile_coating.x, D_profile_coating.data[-1], color=COLORS['blue'])
ax.set_xlim(pebble_radius - interface_half_width, pebble_radius + interface_half_width)
save_fig(fig, "D_interface_concentration.png")

# Report any T concentration values below 1 atom/m3 at the final timestep,
# along with their x location, across both subdomains
print("T concentrations below 1e0 atoms/m3 (final timestep):")
for profile, name in [(T_profile_ceramic, "ceramic"), (T_profile_coating, "coating")]:
    values = np.asarray(profile.data[-1])
    x = np.asarray(profile.x)
    mask = values < 1.0
    for xi, vi in zip(x[mask], values[mask]):
        print(f"  [{name}] x = {xi:.6e} m, T = {vi:.6e} atoms/m3")


# Plot several snapshots in time (log y-axis) to watch the concentration accumulate,
# starting from t_min onwards so the plot focuses on the slower long-term buildup
# rather than the fast initial transient.
def plot_accumulation(profile_inner, profile_outer, species_name, filename, n_snapshots=10, t_min=1000.0):
    t = np.asarray(profile_inner.t)
    start = np.searchsorted(t, t_min)
    n = len(t) - start
    idx = start + np.unique(np.logspace(0, np.log10(n - 1), n_snapshots).astype(int))

    fig, ax = make_fig(
        xlabel="x (m)",
        ylabel=f"{species_name} concentration (m$^{{-3}}$)",
        title=f"{species_name} accumulation over time",
        log_y=True,
    )
    norm = mcolors.LogNorm(vmin=float(t[idx[0]]), vmax=float(t[idx[-1]]))
    cmap = plt.get_cmap(LIGHT_CMAP)
    for i in idx:
        color = cmap(norm(t[i]))
        ax.plot(profile_inner.x, profile_inner.data[i], color=color)
        ax.plot(profile_outer.x, profile_outer.data[i], color=color)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label("Time (s)")
    save_fig(fig, filename)


plot_accumulation(T_profile_ceramic, T_profile_coating, "T", "T_accumulation.png")
plot_accumulation(D_profile_ceramic, D_profile_coating, "D", "D_accumulation.png")

# Plot the tritium surface flux (release rate) vs time, log-log since it spans
# many orders of magnitude while the system approaches steady state
fig, ax = make_fig(
    xlabel="Time (s)",
    ylabel="T surface flux (atoms/m$^2$/s)",
    title="Tritium release rate at the coating surface",
    log_x=True,
    log_y=True,
)
ax.plot(surface_flux.t, surface_flux.data, color=COLORS['red'])
save_fig(fig, "T_surface_flux.png")