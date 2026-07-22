'''
Need to make a script with teh same mesh and some dummy numbers that we know the expected behaviour

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



l2O_festim = F.Material(D_0=1.0, E_D=1.0,
                 K_S_0=1.0, E_K_S=1.0, # placeholder values for solubility, as the Li2O solubility is not well known
                 solubility_law="sievert")
Zr_festim = F.Material(D_0=1.0, E_D=1.0,
                     K_S_0=2.0, E_K_S=2.0,
                     solubility_law="sievert")


# Define mesh
vertices = np.unique(
    np.concatenate(
        [
            np.linspace(
                0.0,
                pebble_radius - local_refinement_thickness,
                num=1500,
            ),
            np.linspace(
                pebble_radius - local_refinement_thickness,
                pebble_radius + local_refinement_thickness,
                num=410,
            ),
            np.linspace(
                pebble_radius + local_refinement_thickness,
                surface_location,
                num=1000,
            ),
            [pebble_radius],
        ]
    )
)

mesh = F.Mesh1D(
    vertices=vertices,
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

# Both BC locations must coincide with an actual mesh vertex, or FESTIM can't
# attach the boundary condition to a DOF there.
assert np.isclose(vertices, 0.0).any(), "x=0.0 is not a mesh vertex"
assert np.isclose(vertices, surface_location).any(), "surface_location is not a mesh vertex"

inner_surface = F.SurfaceSubdomain1D(
    id=5,
    x=0.0,
)

surface = F.SurfaceSubdomain1D(
    id=3,
    x=surface_location,
)

my_model.subdomains = [
    ceramic_subdomain,
    coating_subdomain,
    inner_surface,
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

# Define BCs

left_bc = F.FixedConcentrationBC(
    subdomain=inner_surface,
    value=100.0,
    species=T
)

right_bc = F.FixedConcentrationBC(
    subdomain=surface,
    value=0.0,
    species=T
)
my_model.boundary_conditions = [left_bc, right_bc]


# Define temperature

my_model.temperature = model_temperature # inlet blanket conditions

# Define exports so we can plot the concentration profile at a few snapshots in time
T_profile_ceramic = F.Profile1DExport(field=T, subdomain=ceramic_subdomain)
T_profile_coating = F.Profile1DExport(field=T, subdomain=coating_subdomain)

my_model.exports = [T_profile_ceramic, T_profile_coating]

# Define model settings
# atol scaled for O(1) dummy concentration values (the real model uses O(1e15-1e20)
# atoms/m3, where atol=1e10 makes sense; here it would make the solver treat the
# all-zero initial guess as already converged and never apply the BCs).
my_model.settings = F.Settings(
    atol=1e-6,
    rtol=1e-8,
    max_iterations=500,
    transient=True,
    final_time=1e5,
    stepsize=F.Stepsize(
        initial_value=1e-3,
        growth_factor=1.5,
        cutback_factor=0.5,
        target_nb_iterations=5,
        max_stepsize=1e3,
    ),
)
my_model.initialise()
my_model.run()

# Plot the concentration profile at t = 1, 10, 1000 s (nearest recorded snapshot
# to each, since final_time=1e3 spans that whole range)
from plot_utils import make_fig, save_fig, COLORS

t = np.asarray(T_profile_ceramic.t)
target_times = [1, 10, 1000, 10000, 100000]
palette = [COLORS['red'], COLORS['blue'], COLORS['green'], COLORS['purple'], COLORS['orange']]

fig, ax = make_fig(
    xlabel="x (m)",
    ylabel="T concentration (m$^{-3}$)",
    title="T concentration profile at t = 1, 10, 1000, 10000, 100000 s",
    log_y=True,
)
for target_t, color in zip(target_times, palette):
    idx = int(np.argmin(np.abs(t - target_t)))
    ax.plot(T_profile_ceramic.x, T_profile_ceramic.data[idx], color=color,
            label=f"t = {t[idx]:.3g} s")
    ax.plot(T_profile_coating.x, T_profile_coating.data[idx], color=color)
ax.legend()
save_fig(fig, "dummy_concentration_snapshots_diff_sol.png")

