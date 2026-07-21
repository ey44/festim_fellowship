from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import basix
import numpy as np
import tqdm.autonotebook
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import create_mesh, locate_entities, locate_entities_boundary, meshtags
import ufl

from dolfinx.io import VTXWriter

indices = np.linspace(0, 1, num=100)
gdim = 1
shape = 'interval'
degree = 1

domain = ufl.Mesh(basix.ufl.element("Lagrange", shape, degree, shape=(gdim,)))
mesh_points = np.reshape(indices, (len(indices), 1))
indexes = np.arange(mesh_points.shape[0])
cells = np.stack((indexes[:-1], indexes[1:]), axis=-1)
my_mesh = create_mesh(comm=MPI.COMM_WORLD, cells=cells, x=mesh_points, e=domain)

# Define function space and functions
element_CG = basix.ufl.element(
    basix.ElementFamily.P,
    my_mesh.basix_cell(),
    1,
    basix.LagrangeVariant.equispaced,
)
elements = basix.ufl.mixed_element([element_CG, element_CG])

V = fem.functionspace(my_mesh, elements)
u = fem.Function(V)
u_n = fem.Function(V)

c1, c2 = ufl.split(u)
c1_n, c2_n = ufl.split(u_n)
V1, V2 = V.sub(0), V.sub(1)
v1, v2 = ufl.TestFunction(V)

# Collapsing is required - one gets you a function to write an ouput that pulls the values from the mixed space to the subspace, 
# the other get you the right indexes to map the subspace to the mixed space so you can track the write quantiites
c1_pp, c2_pp = u.sub(0).collapse(), u.sub(1).collapse()
_, map_c1_to_u = V.sub(0).collapse()
_, map_c2_to_u = V.sub(1).collapse()

# Define the boundary conditions
# this is defining the facets, the same as setting up the indexes fo the cells
fdim = my_mesh.topology.dim - 1  # facet dim = one less than cell dim (points, for this 1D mesh)
num_facets = my_mesh.topology.index_map(fdim).size_local  # number of facets on this MPI rank
mesh_facet_indices = np.arange(num_facets, dtype=np.int32)  # an ID for every facet
tags_facets = np.full(num_facets, 0, dtype=np.int32)  # default tag = 0 (untagged) for every facet

#again you want to locate the ones you want to tag that will have BCs
entities_left = locate_entities_boundary(my_mesh, fdim, lambda x: np.isclose(x[0], 0))  # facet(s) at x=0
entities_right = locate_entities_boundary(my_mesh, fdim, lambda x: np.isclose(x[0], 1))  # facet(s) at x=1
tags_facets[entities_left] = 1  # mark left boundary facet with tag 1
tags_facets[entities_right] = 2  # mark right boundary facet with tag 2

# interior flux point: mesh has no exact node at x=0.5, so snap to the nearest actual node coordinate
middle_x = indices[np.argmin(np.abs(indices - 0.5))]
entities_middle = locate_entities(my_mesh, fdim, lambda x: np.isclose(x[0], middle_x))  # interior facet, not a boundary one -> locate_entities, not _boundary
tags_facets[entities_middle] = 3  # mark interior flux facet with tag 3

facet_meshtags = meshtags(my_mesh, fdim, mesh_facet_indices, tags_facets)  # bundle facet indices + tags into a queryable object
# You have to manually create the connectivity between facets and cells, which is needed by locate_dofs_topological to find the DOFs on the facets
# Thsi is key as it is the link between the dimensionality of the mesh and the function space, so you can find the DOFs on the facets
my_mesh.topology.create_connectivity(fdim, my_mesh.topology.dim)  # build facet<->cell lookup, needed by locate_dofs_topological

# Same as before we are now locating which facets we want to tag with boundary conditions
left_facets = facet_meshtags.find(1)  # facet(s) tagged "left"
left_dofs_c1 = fem.locate_dofs_topological(V.sub(0), fdim, left_facets)  # c1 DOFs touching those facets
left_dofs_c2 = fem.locate_dofs_topological(V.sub(1), fdim, left_facets)  # c2 DOFs touching those facets
right_facets = facet_meshtags.find(2)  # facet(s) tagged "right"
right_dofs_c1 = fem.locate_dofs_topological(V.sub(0), fdim, right_facets)
right_dofs_c2 = fem.locate_dofs_topological(V.sub(1), fdim, right_facets)

# Now, finally, we can define the boundary conditions for each component.
bc_left_c1 = fem.dirichletbc(
    fem.Constant(my_mesh, PETSc.ScalarType(100)), left_dofs_c1, V1
)  # c1 = 100 at left boundary
bc_left_c2 = fem.dirichletbc(
    fem.Constant(my_mesh, PETSc.ScalarType(75)), left_dofs_c2, V2
)  # c2 = 75 at left boundary
bc_right_c1 = fem.dirichletbc(fem.Constant(my_mesh, PETSc.ScalarType(0)), right_dofs_c1, V1)  # c1 = 0 at right boundary
bc_right_c2 = fem.dirichletbc(fem.Constant(my_mesh, PETSc.ScalarType(0)), right_dofs_c2, V2)  # c2 = 0 at right boundary
bcs = [bc_left_c1, bc_left_c2, bc_right_c1, bc_right_c2]  # bundle all BCs for the solver

# Define variational problem
k = 0.1
dt = 0.1
# Here you combine them all together into one function that is solved
F = ufl.dot(k * ufl.grad(c1), ufl.grad(v1)) * ufl.dx
F += ufl.dot(k * ufl.grad(c2), ufl.grad(v2)) * ufl.dx
F += ((c1 - c1_n) / dt) * v1 * ufl.dx
F += ((c2 - c2_n) / dt) * v2 * ufl.dx

# interior flux at the middle facet (tag 3): needs the interior facet measure dS (capital S),
# not ds, since this facet is shared by two cells rather than bounding the domain.
# avg(v) is required (not plain v) because test functions have two traces ('+'/'-') at an interior facet.
dS = ufl.Measure("dS", domain=my_mesh, subdomain_data=facet_meshtags)
flux_c1 = 1.0  # magnitude/sign to tune once running - see plan's Verification section
flux_c2 = 1.0
F -= flux_c1 * ufl.avg(v1) * dS(3)
F -= flux_c2 * ufl.avg(v2) * dS(3)

# define solver -- copied from eg_3.py, unchanged
petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_stol": np.sqrt(np.finfo(dolfinx.default_real_type).eps)
    * 1e-2,
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

# CHANGED: output filenames given a "_flux" suffix so they don't overwrite eg_3.py's
# "two_species_c1.bp"/"two_species_c2.bp" output if run from the same directory.
writer1 = VTXWriter(MPI.COMM_WORLD, "two_species_flux_c1.bp", c1_pp, "BP5")
writer2 = VTXWriter(MPI.COMM_WORLD, "two_species_flux_c2.bp", c2_pp, "BP5")

final_time = 10
t = 0
progress = tqdm.autonotebook.tqdm(
        desc="Solving H transport problem", total=final_time, unit_scale=True
    )
while t < final_time:
    solver.solve()

    u_n.x.array[:] = u.x.array

    c1_pp.x.array[:] = u.x.array[map_c1_to_u]
    c2_pp.x.array[:] = u.x.array[map_c2_to_u]

    writer1.write(t)
    writer2.write(t)

    t += dt

    progress.update(dt)

writer1.close()
writer2.close()

# visualise the final state of both species - line plot, since this is a 1D mesh
import matplotlib.pyplot as plt

c1_coords = c1_pp.function_space.tabulate_dof_coordinates()[:, 0]
c2_coords = c2_pp.function_space.tabulate_dof_coordinates()[:, 0]

c1_order = np.argsort(c1_coords)
c2_order = np.argsort(c2_coords)

plt.plot(c1_coords[c1_order], c1_pp.x.array.real[c1_order], marker="o", markersize=3, label="c1")
plt.plot(c2_coords[c2_order], c2_pp.x.array.real[c2_order], marker="o", markersize=3, label="c2")
plt.xlabel("x")
plt.ylabel("concentration")
plt.legend()
plt.show()

