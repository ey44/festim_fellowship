import dolfinx
import numpy as np
import ufl
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import mesh
from dolfinx import fem
import basix

nx = ny = 96

#domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.quadrilateral)
gdim, shape, degree = 2, "quadrilateral", 1
domain_ufl = ufl.Mesh(basix.ufl.element("Lagrange", shape, degree, shape=(gdim,)))
x_coords = np.linspace(0, 1, nx + 1)
y_coords = np.linspace(0, 1, ny + 1)
xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")
mesh_points = np.column_stack((xx.ravel(), yy.ravel()))

def vertex_id(i, j):
    return i * (ny + 1) + j   # matches indexing="ij" layout above

cells = []
for i in range(nx):
    for j in range(ny):
        v0 = vertex_id(i,     j)
        v1 = vertex_id(i + 1, j)
        v2 = vertex_id(i,     j + 1)
        v3 = vertex_id(i + 1, j + 1)
        cells.append([v0, v1, v2, v3])
cells = np.array(cells)

domain = mesh.create_mesh(comm=MPI.COMM_WORLD, cells=cells, x=mesh_points, e=domain_ufl)


# we create a mixed element with two components, both continuous galerkin degree 1
cg_element = basix.ufl.element("Lagrange", domain.basix_cell(), degree=1)

mixed_element = basix.ufl.mixed_element([cg_element, cg_element])

# then we make a functionspace from the mixed element
V = fem.functionspace(domain, mixed_element)

# we create a "main" function u which is a vector of the two components
u = fem.Function(V)

# to use the components in variational forms, we use ufl.split
# the first will be the mobile concentration cm, the second the trapped concentration ct
cm, ct = ufl.split(u)

# we create test functions for both components
v_cm, v_ct = ufl.TestFunctions(V)

'''
dirichlet boundary conditions
'''

def inlet(x):
    return np.logical_and(np.isclose(x[0], 0), x[1] <= 0.5)

def outlet(x):
    return np.logical_and(np.isclose(x[0], 1), x[1] >= 0.5)

V0, submap = V.sub(0).collapse()

# the trick here was to pass both the subspace and the collapsed space to locate_dofs_geometrical
# in FESTIM we don't need this since we use meshtags for everything
# https://fenicsproject.discourse.group/t/dolfinx-dirichlet-bcs-for-mixed-function-spaces/7844/2

dofs_outlet = fem.locate_dofs_geometrical((V.sub(0), V0), outlet)
dofs_inlet = fem.locate_dofs_geometrical((V.sub(0), V0), inlet)

c_inlet = fem.Constant(domain, 1.0)
c_outlet = fem.Constant(domain, 0.0)

bc_outlet = fem.dirichletbc(c_outlet, dofs_outlet[0], V.sub(0))
bc_inlet = fem.dirichletbc(c_inlet, dofs_inlet[0], V.sub(0))


'''
Weak form solution
'''
# Problem parameters
k = 0.1  # trapping rate
p = 0.1  # detrapping rate
n = 1  # total trapping sites
D = 2.0 # diffusion coefficient

trapping = k * cm * (n - ct)
detrapping = p * ct

# NOTE everything is bundled in one variational form F
# the difference between the different equations is made with the test functions v_cm and v_ct
F_mobile = (
    D*ufl.dot(ufl.grad(cm), ufl.grad(v_cm)) * ufl.dx
    - trapping * v_cm * ufl.dx
    + detrapping * v_cm * ufl.dx
)
F_trapped = +trapping * v_ct * ufl.dx - detrapping * v_ct * ufl.dx

F = F_mobile + F_trapped


# taken from https://github.com/FEniCS/dolfinx/blob/5fcb988c5b0f46b8f9183bc844d8f533a2130d6a/python/demo/demo_cahn-hilliard.py#L279C1-L286C28
use_superlu = PETSc.IntType == np.int64  # or PETSc.ScalarType == np.complex64
sys = PETSc.Sys()  # type: ignore
if sys.hasExternalPackage("mumps") and not use_superlu:
    linear_solver = "mumps"
elif sys.hasExternalPackage("superlu_dist"):
    linear_solver = "superlu_dist"
else:
    linear_solver = "petsc"

petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_stol": np.sqrt(np.finfo(dolfinx.default_real_type).eps) * 1e-2,
    "snes_atol": 1e-10,
    "snes_rtol": 1e-10,
    "snes_max_it": 100,
    "snes_divergence_tolerance": "PETSC_UNLIMITED",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": linear_solver,
}

problem = NonlinearProblem(
    F,
    u,
    bcs=[bc_outlet, bc_inlet],
    petsc_options=petsc_options,
    petsc_options_prefix="Poisson",
)

problem.solve()

print("SOLVER CONVERGED")


'''
Plotting
'''
# we first split the main solution u into its components with .split()
cm_post, ct_post = u.split()  # NOTE this is different from ufl.split(u)

# for postprocessing, it's easier to work with collapsed functions
cm_post = cm_post.collapse()
ct_post = ct_post.collapse()

import pyvista
from dolfinx import plot

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(tdim, tdim)

# We can do this once since both components share the same function space
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(cm_post.function_space)


plotter = pyvista.Plotter(shape=(1, 2))


plotter.subplot(0, 0)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["cm"] = cm_post.x.array.real
u_grid.set_active_scalars("cm")
plotter.add_mesh(u_grid, show_edges=False)
plotter.view_xy()

plotter.subplot(0, 1)
ct_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
ct_grid.point_data["ct"] = ct_post.x.array.real
ct_grid.set_active_scalars("ct")
plotter.add_mesh(ct_grid, show_edges=False)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()