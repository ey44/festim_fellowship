from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import basix
import numpy as np
import tqdm.autonotebook
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import create_mesh
import ufl
from dolfinx import plot
import pyvista

from dolfinx.io import VTXWriter


def plot_function(function):

    topology, cell_types, geometry = plot.vtk_mesh(function.function_space)
    u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    u_grid.point_data["c"] = function.x.array.real
    u_grid.set_active_scalars("c")
    u_plotter = pyvista.Plotter()

    u_plotter.add_mesh(u_grid, cmap="viridis", show_edges=False)

    u_plotter.show()


# define mesh
indices = np.linspace(0, 1, num=100)
gdim, shape, degree = 1, "interval", 1
domain = ufl.Mesh(basix.ufl.element("Lagrange", shape, degree, shape=(gdim,)))
mesh_points = np.reshape(indices, (len(indices), 1))
indexes = np.arange(mesh_points.shape[0])
cells = np.stack((indexes[:-1], indexes[1:]), axis=-1)
my_mesh = create_mesh(comm=MPI.COMM_WORLD, cells=cells, x=mesh_points, e=domain)
fdim = my_mesh.topology.dim - 1

# Define function space and functions
V = fem.functionspace(my_mesh, ("Lagrange", 1))
u = fem.Function(V)
u_n = fem.Function(V)
v = ufl.TestFunction(V)

# define boundary conditions
dofs_L = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
dofs_R = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], indices[-1]))
bc_left = fem.dirichletbc(fem.Constant(my_mesh, PETSc.ScalarType(100)), dofs_L, V)
bc_right = fem.dirichletbc(fem.Constant(my_mesh, PETSc.ScalarType(0)), dofs_R, V)
bcs = [bc_left, bc_right]

# Define variational problem
k = 0.1
dt = 0.1
F = ufl.dot(k * ufl.grad(u), ufl.grad(v)) * ufl.dx
F += ((u - u_n) / dt) * v * ufl.dx

# define solver
petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_stol": np.sqrt(np.finfo(dolfinx.default_real_type).eps) * 1e-2,
    "snes_atol": 1e-10,
    "snes_rtol": 1e-10,
    "snes_max_it": 30,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
solver = NonlinearProblem(
    F,
    u,
    bcs=bcs,
    petsc_options=petsc_options,
    petsc_options_prefix="festim_solver",
)
snes = solver.solver
prefix = snes.getOptionsPrefix()
opts = PETSc.Options()
for k in petsc_options.keys():
    del opts[f"{prefix}{k}"]

writer = VTXWriter(MPI.COMM_WORLD, "ht_transient", u, "BP5")

final_time = 10
t = 0
progress = tqdm.autonotebook.tqdm(
    desc="Solving H transport problem", total=final_time, unit_scale=True
)
while t < final_time:
    solver.solve()
    u_n.x.array[:] = u.x.array
    writer.write(t)

    t += dt

    progress.update(dt)

writer.close()
