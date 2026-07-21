import festim as F
import ufl

T0 = 300
my_model = F.HydrogenTransportProblem()
my_model.temperature = lambda x: T0 * ufl.exp(-x[0])  


# my_model.temperature = lambda t: T0 * ufl.sin(t)
# my_model.temperature = lambda x: T0 * (ufl.cos(x[0])*ufl.sin(x[1]) - 2*x[2])


from dolfinx.mesh import create_unit_cube
from mpi4py import MPI
import ufl
import festim as F
import numpy as np

mesh = F.Mesh(create_unit_cube(MPI.COMM_WORLD, 10, 10, 10))
my_model.mesh = mesh

mat = F.Material(D_0=1, E_D=0)

volume = F.VolumeSubdomain(id=1, material=mat)
top_surface = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[2], 1.0))
bottom_surface = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[2], 0.0))
my_model.subdomains = [top_surface, bottom_surface, volume]

H = F.Species("H")
my_model.species = [H]

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=top_surface, value=1.0, species=H),
    F.FixedConcentrationBC(subdomain=bottom_surface, value=0.0, species=H),
]

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

my_model.initialise()
my_model.run()


print("Temperature at (0, 0, 0):", my_model.temperature(np.array([0.0, 0.0, 0.0])))



print("RUNNING AN ADVANCED SIMULATION")

import festim as F

from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import NonlinearProblem

heat_transfer_model = F.HeatTransferProblem()

def thermal_conductivity_material(x):
    return 3 + 0.1 * x

mat = F.Material(D_0=1, E_D=0, thermal_conductivity=thermal_conductivity_material) #

import dolfinx
from mpi4py import MPI
import numpy as np

nx = ny = 20

mesh_fenics = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)

domain = dmesh.create_unit_square(MPI.COMM_WORLD, nx, ny) # making a side by side dofinx version
# Functions space for dolfinx
V = fem.functionspace(domain, ("Lagrange", 1))
T = fem.Function(V)
v = ufl.TestFunction(V)

heat_transfer_model.mesh = F.Mesh(mesh_fenics)


'''
Domains and boundary conditions
'''
# FESTIM
volume_subdomain = F.VolumeSubdomain(id=1, material=mat)

top_bot = F.SurfaceSubdomain(id=2, locator=lambda x: np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)))
left = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], 0.0))
right = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[0], 1.0))


heat_transfer_model.subdomains = [volume_subdomain, top_bot, left, right]

heat_transfer_model.sources = [
    F.HeatSource(value=lambda x: 1 + 0.1 * x[0], volume=volume_subdomain)
]

import ufl

fixed_temperature_left = F.FixedTemperatureBC(
    subdomain=left, value=lambda x: 350 + 20 * ufl.cos(x[0]) * ufl.sin(x[1])
)

def h_coeff(x):
    return 100 * x[0]

def T_ext(x):
    return 300 + 3 * x[1]

convective_heat_transfer = F.HeatFluxBC(
    subdomain=top_bot, value=lambda x, T: h_coeff(x) * (T_ext(x) - T)
)

heat_flux = F.HeatFluxBC(
    subdomain=right, value=lambda x: 10 + 3 * ufl.cos(x[0]) + ufl.sin(x[1])
)

heat_transfer_model.boundary_conditions = [
    fixed_temperature_left,
    convective_heat_transfer,
    heat_flux,
]
# DOLPHINX
fdim = domain.topology.dim - 1
facets_left = dmesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
facets_right = dmesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 1.0))
facets_topbot = dmesh.locate_entities_boundary(domain, fdim, lambda x: np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)))

indices = np.concatenate([facets_left, facets_right, facets_topbot])
values = np.concatenate([
    np.full_like(facets_left, 1),
    np.full_like(facets_right, 2),
    np.full_like(facets_topbot, 3),
])
order = np.argsort(indices)
facet_tags = dmesh.meshtags(domain, fdim, indices[order], values[order])
domain.topology.create_connectivity(fdim, domain.topology.dim)

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)  # exterior facet measure - same as our Neumann discussion earlier

left_dofs = fem.locate_dofs_topological(V, fdim, facets_left)
x = ufl.SpatialCoordinate(domain)
T_left_expr = fem.Expression(350 + 20*ufl.cos(x[0])*ufl.sin(x[1]), V.element.interpolation_points)
T_left = fem.Function(V)
T_left.interpolate(T_left_expr)
bc_left = fem.dirichletbc(T_left, left_dofs)
bcs = [bc_left]

lam = 3 + 0.1 * T   # lambda(T), a UFL expression depending on the unknown T itself - same nonlinear-coefficient trick as D_eff(c_hyd) in eg_4.py

Q = 1 + 0.1 * x[0]                       # volume heat source
q_right = 10 + 3*ufl.cos(x[0]) + ufl.sin(x[1])   # prescribed Neumann flux on the right
h_val = 100 * x[0]      # NOTE: named h_val, not h - avoids shadowing anything (h wasn't actually taken, renamed defensively)
T_ext_val = 300 + 3*x[1]  # NOTE: named T_ext_val, not T_ext - "T_ext" is already taken by the def T_ext(x) function above, which FESTIM calls lazily during initialise(), so overwriting it here broke that BC
# Robin/convective term on top+bottom: q_n = h*(T_ext - T) - depends on T itself, like our earlier Robin BC discussion

weak_form = ufl.dot(lam * ufl.grad(T), ufl.grad(v)) * ufl.dx  # NOTE: named weak_form, not F - "F" is already taken by `import festim as F` in this script
weak_form -= Q * v * ufl.dx
weak_form -= q_right * v * ds(2)                         # right boundary: pure Neumann, doesn't touch the Jacobian
weak_form -= h_val * (T_ext_val - T) * v * ds(3)          # top/bottom: Robin, DOES touch the Jacobian since it depends on T





# FESTIM
heat_transfer_model.settings = F.Settings(
    transient=False,
    atol=1e-09,
    rtol=1e-09,
)

heat_transfer_model.initialise()
heat_transfer_model.run()

# DOLFINX
petsc_options = {
    "snes_type": "newtonls",
    "snes_atol": 1e-9,
    "snes_rtol": 1e-9,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
solver = NonlinearProblem(weak_form, T, bcs=bcs, petsc_options=petsc_options, petsc_options_prefix="heat_solver")
snes = solver.solver
prefix = snes.getOptionsPrefix()
opts = PETSc.Options()
for k in petsc_options.keys():
    del opts[f"{prefix}{k}"]

solver.solve()




# side-by-side comparison: FESTIM's solve vs our raw-dolfinx solve, same problem, same mesh params
# named separately (festim_T) rather than reusing T - reusing T here was the same class of bug as the
# F/T_ext collisions above, since it silently clobbered our own dolfinx solution's Function object
festim_T = heat_transfer_model.u
festim_T_coords = festim_T.function_space.tabulate_dof_coordinates()
festim_T_values = festim_T.x.array.real

dolfinx_T_coords = V.tabulate_dof_coordinates()  # our own raw-dolfinx T, solved earlier, untouched by the FESTIM run
dolfinx_T_values = T.x.array.real

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

im0 = axes[0].tricontourf(festim_T_coords[:, 0], festim_T_coords[:, 1], festim_T_values, levels=20, cmap="inferno")
axes[0].set_title("FESTIM")
axes[0].set_aspect("equal")
fig.colorbar(im0, ax=axes[0], label="Temperature")

im1 = axes[1].tricontourf(dolfinx_T_coords[:, 0], dolfinx_T_coords[:, 1], dolfinx_T_values, levels=20, cmap="inferno")
axes[1].set_title("Raw dolfinx")
axes[1].set_aspect("equal")
fig.colorbar(im1, ax=axes[1], label="Temperature")

plt.tight_layout()
plt.show()

# Remi's suggestion for calculation of residual difference $\sqrt(\int (u - u_ref)^2 dx)$
def remi_fun(u, u_ref):
    for each in enumerate(u):
        u[each] = u_ref[each]
    
    np.sqrt(np.sum((u - u_ref) ** 2))

remi_fun_T_values = remi_fun(festim_T_values, dolfinx_T_values)

ax = plt.figure(figsize=(12, 5))

im3 = ax.tricontourf(festim_T_coords[:, 0], festim_T_coords[:, 1], remi_fun_T_values, levels=20, cmap="inferno")
ax.set_title("Residual difference between FESTIM and raw dolfinx")
fig.colorbar(im3, ax=ax, label="Temperature")

plt.tight_layout()
plt.show()

'''
Linking it with hydrogen transport simulation
'''


hydrogen_problem = F.HydrogenTransportProblem()
hydrogen_problem.mesh = heat_transfer_model.mesh  
H = F.Species("H")
hydrogen_problem.species = [H]
hydrogen_problem.temperature = heat_transfer_model.u

hydrogen_problem.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=left, value=1.0, species=H),
    F.FixedConcentrationBC(subdomain=right, value=0.0, species=H)
]

hydrogen_problem.subdomains = heat_transfer_model.subdomains

hydrogen_problem.settings = F.Settings(
    transient=False,
    atol=1e-09,
    rtol=1e-09,
)  

hydrogen_problem.initialise()
hydrogen_problem.run()

import pyvista
from dolfinx import plot

pyvista.set_jupyter_backend("html")

c = H.post_processing_solution  # FIXED: was hydrogen_problem.u - that's the raw mixed-element wrapper (festim wraps even single-species problems in a mixed element), which plot.vtk_mesh can't handle directly. post_processing_solution is the already-collapsed, standalone version - same idea as species.ipynb's plot_profile

topology, cell_types, geometry = plot.vtk_mesh(c.function_space)  # FIXED: was T.function_space - T no longer points to heat_transfer_model.u now that festim_T has its own name, and c has its own function_space anyway
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["c"] = c.x.array.real
u_grid.set_active_scalars("c")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, cmap="viridis", show_edges=False)

u_plotter.view_xy()

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")  # FIXED: was "temperature.png" - wrong filename, this is the concentration plot



