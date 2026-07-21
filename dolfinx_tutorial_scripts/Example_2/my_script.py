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

from dolfinx.io import VTXWriter

indices = np.linspace(0, 1, num=100)

gdim = 1
shape = "interval"
degree = 1 

domain = ufl.Mesh(basix.ufl.element("Lagrange", shape, degree, shape=(gdim,))) 
mesh_points = np.reshape(indices, (len(indices), 1))
indexes = np.arange(mesh_points.shape[0])
cells = np.stack((indexes[:-1], indexes[1:]), axis=-1)
my_mesh = create_mesh(comm=MPI.COMM_WORLD, cells=cells, x=mesh_points, e=domain)

# Define function space and functions
V = fem.functionspace(my_mesh, ("Lagrange", 1)) # this builds the function space where teh solution will be 
u = fem.Function(V) # this the function you are solving for, and the current time steps solution
u_n = fem.Function(V) # this is the function that will hold the solution from the previous time step, so we can use it in the next time step
v = ufl.TestFunction(V) # this is the test function, which is used in the weak formulation of the problem, we pass in the function space we just created

# define boundary conditions
dofs_L = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
dofs_R = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], indices[-1]))
bc_left = fem.dirichletbc(fem.Constant(my_mesh, PETSc.ScalarType(100)), dofs_L, V)
bc_right = fem.dirichletbc(fem.Constant(my_mesh, PETSc.ScalarType(0)), dofs_R, V)
bcs = [bc_left, bc_right]

# Define variational problem
k = 0.1
dt = 0.1
half_life = 12.32 * 365.25 * 24 * 3600  # tritium half-life in seconds
lam = np.log(2) / half_life

F = D * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx      # diffusion (existing term)
F += ((u - u_n) / dt) * v * ufl.dx                        # time derivative (existing term)
F += lam * u * v * ufl.dx                                 # NEW: decay sink

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








