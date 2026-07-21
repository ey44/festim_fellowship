from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import basix
import numpy as np
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import create_mesh
import ufl

import matplotlib.pyplot as plt


def plot_function(function):
    V = function.function_space
    coords = V.tabulate_dof_coordinates()[:, 0]
    values = function.x.array.real

    order = np.argsort(coords)
    plt.plot(coords[order], values[order], marker="o", markersize=3)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.show()


### write your script here ###
'''
Create deatils on the mesh
'''
points = np.linspace(0, 1, num=100)
gdim = 1
shape = "interval"

mesh_metadata = basix.ufl.element("Lagrange", shape, 1, shape=(gdim,)) # element family, shape of the cell, degree [so could be a higher degree for the FE], then the shape of the element e.g. an x coordinate, or x,y in 2D
mesh_points = np.reshape(points, (len(points), 1)) # dolphinx expects the points to be in a 2D array, so we reshape the 1D array of points into a 2D array with one column
indexes = np.arange(mesh_points.shape[0]) # this gives us indexes for the points
cells = np.stack((indexes[:-1], indexes[1:]), axis=-1) # this gives us the cells, which are the intervals between the points, so we stack the indexes of the points to get the cells
my_mesh = create_mesh(comm=MPI.COMM_WORLD, cells=cells, x=mesh_points, e=mesh_metadata) # this creates the mesh, we pass in the communicator, the cells, the points, and the mesh metadata which expalin how to interprety the rest

'''
We have now created the mesh, where we gave it each set of cell and the points on the mesh, with an explainer at the end

We now have to explain that there is going ot be a function solved across the mesh
'''

variable = fem.functionspace(my_mesh, ("Lagrange", 1)) # this creates a function space, which is where the solution will be defined, we pass in the mesh and the type of function space we want to use, in this case a Lagrange function space of degree 1 which is the same as how we defined in my_mesh
function_skeleton = fem.Function(variable) # this creates a function, which is where the solution will be stored, we pass in the function space we just created
test_function_v = ufl.TestFunction(variable) # this creates a test function, which is used in the variational formulation of the problem, we pass in the function space we just created

# define boundary conditions
dofs_L = fem.locate_dofs_geometrical(variable, lambda x: np.isclose(x[0], 0)) # finds where it is -
dofs_R = fem.locate_dofs_geometrical(variable, lambda x: np.isclose(x[0], points[-1]))
bc_left = fem.dirichletbc(fem.Constant(my_mesh, PETSc.ScalarType(100)), dofs_L, variable) # 
bc_right = fem.dirichletbc(fem.Constant(my_mesh, PETSc.ScalarType(0)), dofs_R, variable)
bcs = [bc_left, bc_right]

# Define the problem
k = 0.1
F = ufl.dot(k * ufl.grad(function_skeleton), ufl.grad(test_function_v)) * ufl.dx

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
    F, # the residual form, which is the variational formulation of the problem
    function_skeleton, # 
    bcs=bcs,
    petsc_options=petsc_options,
    petsc_options_prefix="festim_solver",
)
snes = solver.solver
prefix = snes.getOptionsPrefix()
opts = PETSc.Options()
for k in petsc_options.keys():
    del opts[f"{prefix}{k}"]

# solve problem
solver.solve()

plot_function(function_skeleton)




