"""
creating a smudged analysis of a 2D packed bed

The idea is to have an advection-diffusion problem
with the pebbles creating a source term - initially uniform across the material, but later being a funciton of location temperaeture and partial pressure



"""

import os

import numpy as np
import dolfinx
from dolfinx import plot
from mpi4py import MPI
import basix
from dolfinx import fem
import math as m
import pyvista

script_dir = os.path.dirname(os.path.abspath(__file__))
import festim as F


# Bed parameters

Lx = 0.2  # m
Ly = 0.2  # m
nx = 100
ny = 100

porosity = 0.4
pebble_space = 1 - porosity

model_temperature = 573  # K

# Gas properties
D_He = 1e-4
tau_geom = 1.5  # toruosity

D_eff = D_He * porosity / tau_geom

# Purge flow velocity estimation - taken from calculated interstital velocity
diameter_tube = 4  # m
cross_section = m.pi * (diameter_tube / 2) ** 2  # assumed a tube for now

pressure = 2e5  # Pa
molecular_flow_rate = 250  # mol/s
R_GAS = 8.314  # J/mol/K
volumetric_flow_rate = (
    molecular_flow_rate * R_GAS * model_temperature / pressure
)  # m^3/s, ideal gas law
print(f"Volumetric flow rate: \t {volumetric_flow_rate} \t m^3/s")

v_superficial = volumetric_flow_rate / cross_section
print(f"Superficial velocity: \t {v_superficial} \t m/s")

# 3. Calculate interstitial velocity (v_int = v_s / porosity)
v_interstitial = v_superficial / porosity
print(f"Interstitial velocity: \t {v_interstitial} \t m/s")

# Meshing
dolfinx_mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD,
    points=[np.array([0.0, 0.0]), np.array([Lx, Ly])],
    n=[nx, ny],
    cell_type=dolfinx.mesh.CellType.triangle,
)

print("\nMesh Created")

my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh(mesh=dolfinx_mesh, coordinate_system="cartesian")

# You could split here, make pebble subdomains etc. but here we want to make a big old one

bed_gas = F.Material(
    D_0=D_eff, E_D=0.0
)  # D_0 already the effective value; E_D=0 -> D=D_eff

bed_subdomain = F.VolumeSubdomain(
    id=1,
    material=bed_gas,
    locator=lambda x: np.full(x.shape[1], True),  # whole domain
)

# Surface subdomains (facets) for BCs.
inlet = F.SurfaceSubdomain(
    id=2, locator=lambda x: np.isclose(x[0], 0.0)
)  # x = 0, flow enters
outlet = F.SurfaceSubdomain(
    id=3, locator=lambda x: np.isclose(x[0], Lx)
)  # x = L_x, flow leaves

my_model.subdomains = [bed_subdomain, inlet, outlet]

T_gas = F.Species(name="T_gas")
T_solid = F.Species(
    name="T_solid", mobile=False
)  # immobile, only diffuses through the solid

my_model.species = [T_gas, T_solid]

"""
You coudl create a velocity field here:
from basix.ufl import element

el = element("Lagrange", mesh_fenics.topology.cell_name(), 2, shape=(mesh_fenics.geometry.dim, ))


V = dolfinx.fem.functionspace(my_model.mesh.mesh, el)

velocity = dolfinx.fem.Function(V)

velocity.interpolate(lambda x: (-100*x[1]*(x[1]-1), np.full_like(x[0], 0.0)))

but for now we use a single element, with a steady v throughout i.e. plug flow
"""
v_elem = basix.ufl.element(
    "Lagrange",
    dolfinx_mesh.topology.cell_name(),
    1,
    shape=(dolfinx_mesh.topology.dim,),
)  # family, cell type, degree


V_vel = fem.functionspace(dolfinx_mesh, v_elem)
velocity = fem.Function(V_vel)  # funciton to mape to


velocity.interpolate(
    lambda x: np.vstack([np.full(x.shape[1], v_interstitial), np.zeros(x.shape[1])])
)  # need to have shape for the x and y components

my_model.advection_terms = [
    F.AdvectionTerm(velocity=velocity, subdomain=bed_subdomain, species=T_gas),
]

"""

# plot the field
topology, cell_types, geometry = plot.vtk_mesh(V_vel)
values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
values[:, :len(velocity)] = velocity.x.array.real.reshape((geometry.shape[0], len(velocity)))

# Create a point cloud of glyphs
function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
function_grid["v"] = values
glyphs = function_grid.glyph(orient="v", factor=0.005)

# Create a pyvista-grid for the mesh
dolfinx_mesh.topology.create_connectivity(dolfinx_mesh.topology.dim, dolfinx_mesh.topology.dim)
grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(dolfinx_mesh, dolfinx_mesh.topology.dim))

# Create plotter
plotter = pyvista.Plotter()
plotter.add_mesh(grid, style="wireframe", color="k")
plotter.add_mesh(glyphs)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    fig_as_array = plotter.screenshot("glyphs.png")
"""

# Set up the boundary conditions
inlet_bc = F.FixedConcentrationBC(subdomain=inlet, value=0.0, species=T_gas)
my_model.boundary_conditions = [inlet_bc]


# Defint source
# volumetric tritium generation rate S [particles/m3/s]
P_fus = 1140  # MW
TBR = 1.2
space_volume = Lx * Ly * Lx  # m3
pebble_radius = 1e-3


pebbles_volume = space_volume * (1 - porosity)
tritium_production_rate = P_fus * TBR * 6.2415 * 10**24 / 17.6e6  # tritium atoms/s

volume_per_pebble = (4 / 3) * np.pi * pebble_radius**3
number_of_pebbles = pebbles_volume / volume_per_pebble

# volume_per_pebble cancels out here: S is a density (atoms/m3/s) applied uniformly
# across the ceramic subdomain, so it's just total production over total breeder volume,
# independent of how many pebbles that volume is divided into.
S = tritium_production_rate / pebbles_volume


def pebble_source(x):
    """Spatially varying source term (UFL, symbolic x)."""
    return S + 0.0 * x[0]  # constant 1 everywhere
    # spatial examples:
    # return S0 * ufl.exp(-x[0] / L_x)             # decays along the flow
    # return ufl.conditional(x[0] < 0.1, S0, 0.0)  # source only near inlet


my_model.sources = [
    F.ParticleSource(value=pebble_source, volume=bed_subdomain, species=T_solid),
]

my_model.temperature = model_temperature  # uniform for now


my_reaction = F.Reaction(
    reactant=[T_solid],
    product=[T_gas],
    k_0=1e-12,
    E_k=0.0,
    p_0=0.0,
    E_p=0.0,
    volume=bed_subdomain,
)
my_model.reactions = [my_reaction]


# Exports
c_vtx = F.VTXSpeciesExport(
    filename=os.path.join(script_dir, "bed_T_both.bp"),
    field=[T_gas, T_solid],
    subdomain=bed_subdomain,
)
outlet_flux = F.SurfaceFlux(field=T_gas, surface=outlet)
solid_inv = F.TotalVolume(field=T_solid, volume=bed_subdomain)

my_model.exports = [c_vtx, outlet_flux, solid_inv]


flow_residence_time = Lx / v_interstitial  # time for gas to cross the bed
# final_time = 200.0 * flow_residence_time
final_time = 1

my_model.settings = F.Settings(
    atol=1e8,
    rtol=1e-8,
    max_iterations=30,
    transient=True,
    final_time=final_time,
    stepsize=F.Stepsize(initial_value=0.01, growth_factor=1.1, target_nb_iterations=5),
)

my_model.initialise()
my_model.run()

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 16})

plt.figure(figsize=(12, 9))
plt.plot(solid_inv.t, solid_inv.data)

plt.xlim(left=0)
plt.ylim(bottom=0)
plt.title("T_solid Inventory (T/m)", loc="left")
plt.xlabel("Time (s)")
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()

plt.figure(figsize=(12, 9))
plt.plot(outlet_flux.t, np.array(outlet_flux.data) * -1)

plt.xlim(left=0)
plt.ylim(bottom=0)
plt.xlabel("Time (s)")
plt.title("Outlet Surface Flux (T/m/s)", loc="left")
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()

plt.show()

# Plot the final T concentration field over the bed
# V_c = T.post_processing_solution.function_space
# topology, cell_types, geometry = plot.vtk_mesh(V_c)
# grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# grid["T"] = T.post_processing_solution.x.array.real

# plotter = pyvista.Plotter(off_screen=True)
# plotter.add_mesh(grid, scalars="T", cmap="viridis")
# plotter.view_xy()
# plotter.screenshot(os.path.join(script_dir, "bed_T_field.png"))
# print("Saved: bed_T_field.png")

# # Plot the outlet flux vs time
# import matplotlib.pyplot as plt

# fig, ax = plt.subplots()
# ax.plot(outlet_flux.t, outlet_flux.data)
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("T outlet flux (atoms/m$^2$/s)")
# ax.set_title("Tritium flux at bed outlet")
# fig.savefig(os.path.join(script_dir, "bed_outlet_flux.png"))
# print("Saved: bed_outlet_flux.png")
