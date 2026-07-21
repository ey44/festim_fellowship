from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import basix
import numpy as np
import tqdm.autonotebook
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import create_mesh, locate_entities_boundary, meshtags
import ufl

from dolfinx.io import VTXWriter

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
element_CG_cm = basix.ufl.element(
    basix.ElementFamily.P,
    my_mesh.basix_cell(),
    1,
    basix.LagrangeVariant.equispaced,
)
element_DG_ct = basix.ufl.element(
    "DG",
    my_mesh.basix_cell(),
    1,
    basix.LagrangeVariant.equispaced,
)
element_CG_O = basix.ufl.element(basix.ElementFamily.P, 
                                 my_mesh.basix_cell(), 
                                 1, 
                                 basix.LagrangeVariant.equispaced)
element_CG_H2O = basix.ufl.element(basix.ElementFamily.P,
                                 my_mesh.basix_cell(),
                                 1,
                                 basix.LagrangeVariant.equispaced)
# hydride phase: DG, like ct - it's immobile once formed, not diffusing like cm/c_o
element_DG_hyd = basix.ufl.element("DG", my_mesh.basix_cell(), 1, basix.LagrangeVariant.equispaced)

elements = basix.ufl.mixed_element([element_CG_cm, element_DG_ct, element_CG_O, element_CG_H2O, element_DG_hyd])

V = fem.functionspace(my_mesh, elements)
u = fem.Function(V)
u_n = fem.Function(V)

cm, ct, c_o, c_h2O, c_hyd = ufl.split(u)
cm_n, ct_n, c_o_n, c_h2O_n, c_hyd_n = ufl.split(u_n)
V1, V2, V3, V4, V5 = V.sub(0), V.sub(1), V.sub(2), V.sub(3), V.sub(4)
v_cm, v_ct, v_o, v_h2O, v_hyd = ufl.TestFunction(V)
cm_pp, ct_pp, o_pp, h2o_pp, hyd_pp = u.sub(0).collapse(), u.sub(1).collapse(), u.sub(2).collapse(), u.sub(3).collapse(), u.sub(4).collapse()

_, map_cm_to_u = V.sub(0). collapse()
_, map_ct_to_u = V.sub(1). collapse()
_, map_o_to_u = V.sub(2). collapse()
_, map_h2o_to_u = V.sub(3). collapse()
_, map_hyd_to_u = V.sub(4).collapse()

# define boundary conditions
num_facets = my_mesh.topology.index_map(fdim).size_local
mesh_facet_indices = np.arange(num_facets, dtype=np.int32)
tags_facets = np.full(num_facets, 0, dtype=np.int32)

entities_left = locate_entities_boundary(my_mesh, fdim, lambda x: np.isclose(x[0], 0))
entities_right = locate_entities_boundary(my_mesh, fdim, lambda x: np.isclose(x[0], 1))
tags_facets[entities_left] = 1
tags_facets[entities_right] = 2

facet_meshtags = meshtags(my_mesh, fdim, mesh_facet_indices, tags_facets)
my_mesh.topology.create_connectivity(fdim, my_mesh.topology.dim)

left_facets = facet_meshtags.find(1)
left_dofs_c1 = fem.locate_dofs_topological(V.sub(0), fdim, left_facets)
right_facets = facet_meshtags.find(2)
right_dofs_c1 = fem.locate_dofs_topological(V.sub(0), fdim, right_facets)

bc_left_cm = fem.dirichletbc(
    fem.Constant(my_mesh, PETSc.ScalarType(100)), left_dofs_c1, V1
)
bc_right_cm = fem.dirichletbc(fem.Constant(my_mesh, PETSc.ScalarType(0)), right_dofs_c1, V1)

left_dofs_O = fem.locate_dofs_topological(V.sub(2), fdim, left_facets)
right_dofs_O = fem.locate_dofs_topological(V.sub(2), fdim, right_facets)

bc_left_O = fem.dirichletbc(fem.Constant(my_mesh, PETSc.ScalarType(10.0)), left_dofs_O, V3)   # trace inflow, note: 1.0 vs cm's 100
bc_right_O = fem.dirichletbc(fem.Constant(my_mesh, PETSc.ScalarType(0.0)), right_dofs_O, V3)
bcs = [bc_left_cm, bc_right_cm, bc_left_O, bc_right_O]


# Define variational problem
D_pristine = 0.1   # was: D = 0.1
D_hydride = 0.01   # diffusivity through fully hydride-formed material - 10x slower, tune as desired
c_scale = 10.0     # half-saturation constant: the c_hyd value at which D is halfway between D_pristine and D_hydride
dt = 0.05
k = 1.0
p = 0.001
n = 1
D_O = 0.01       # oxygen's own diffusivity, can differ from D
k_HO = 1e-4      # very small reaction rate -> reacts only slightly

k_hyd = 0.05     # hydride formation rate
p_hyd = 1e-6     # hydride dissociation rate - near zero: unlike trapping's p, hydride is effectively irreversible once formed

reaction_HO = -k_HO * cm * c_o   # FIXED: pre-negated to match reaction/reaction_hyd's convention -
                                  # was previously unnegated, which made cm/c_o gain and c_h2O lose
                                  # material (backwards), producing unphysical negative c_h2O values.
reaction = -k * cm * (n - ct) + p * ct
# no (n - c_hyd) capacity term here: hydride formation isn't limited to a fixed number of
# discrete trap sites the way ct is - it's a new phase, so just proportional to cm itself
reaction_hyd = -k_hyd * cm + p_hyd * c_hyd

# D_eff: cm's diffusivity now depends on c_hyd (another unknown in this same coupled system).
# c_hyd/(c_hyd+c_scale) is a smooth, saturating 0->1 blend (Michaelis-Menten style) - no threshold,
# no ufl.conditional, so the Jacobian stays smooth/differentiable everywhere (good for Newton).
# 0 when c_hyd=0 (pristine diffusivity) -> approaches 1 as c_hyd grows (approaches D_hydride).
D_eff = D_pristine - (D_pristine - D_hydride) * (c_hyd / (c_hyd + c_scale))

# cm now loses material to trapping, the oxygen reaction, AND hydride formation
F = ufl.dot(D_eff * ufl.grad(cm), ufl.grad(v_cm)) * ufl.dx - reaction * v_cm * ufl.dx - reaction_HO * v_cm * ufl.dx - reaction_hyd * v_cm * ufl.dx
F += ((cm - cm_n) / dt) * v_cm * ufl.dx

F += reaction * v_ct * ufl.dx
F += ((ct - ct_n) / dt) * v_ct * ufl.dx

# oxygen: diffuses, and is consumed by the same reaction
F += ufl.dot(D_O * ufl.grad(c_o), ufl.grad(v_o)) * ufl.dx - reaction_HO * v_o * ufl.dx  # FIXED: v_c_o -> v_o (matches ufl.TestFunction name)
F += ((c_o - c_o_n) / dt) * v_o * ufl.dx  # FIXED: v_c_o -> v_o

# FIXED: c_h2O was declared in the mixed space but had zero terms in F, giving the assembled
# Jacobian a zero row/column for those DOFs -> singular matrix -> LU/MUMPS factorization would fail.
# Wired in as the reaction product: gains exactly what cm/c_o lose to reaction_HO, no diffusion
# term (treated as a local, non-diffusing product, same reasoning as ct being DG).
F += reaction_HO * v_h2O * ufl.dx
F += ((c_h2O - c_h2O_n) / dt) * v_h2O * ufl.dx

# hydride: no diffusion term (immobile solid phase, DG element) - "different properties" from cm
# (mobile) and ct (a reversible trap): unlimited capacity, near-irreversible once formed.
F += reaction_hyd * v_hyd * ufl.dx
F += ((c_hyd - c_hyd_n) / dt) * v_hyd * ufl.dx


# define solver
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

writer1 = VTXWriter(MPI.COMM_WORLD, "trapping_cm.bp", cm_pp, "BP5")
writer2 = VTXWriter(MPI.COMM_WORLD, "trapping_ct.bp", ct_pp, "BP5")
writer3 = VTXWriter(MPI.COMM_WORLD, "trapping_o.bp", o_pp, "BP5")
writer4 = VTXWriter(MPI.COMM_WORLD, "trapping_h2o.bp", h2o_pp, "BP5")
writer5 = VTXWriter(MPI.COMM_WORLD, "trapping_hyd.bp", hyd_pp, "BP5")

final_time = 10
t = 0
progress = tqdm.autonotebook.tqdm(
        desc="Solving H transport problem", total=final_time, unit_scale=True
    )
while t < final_time:
    solver.solve()

    u_n.x.array[:] = u.x.array

    cm_pp.x.array[:] = u.x.array[map_cm_to_u]
    ct_pp.x.array[:] = u.x.array[map_ct_to_u]
    o_pp.x.array[:] = u.x.array[map_o_to_u]
    h2o_pp.x.array[:] = u.x.array[map_h2o_to_u]
    hyd_pp.x.array[:] = u.x.array[map_hyd_to_u]

    writer1.write(t)
    writer2.write(t)
    writer3.write(t)
    writer4.write(t)
    writer5.write(t)

    t += dt

    progress.update(dt)

writer1.close()
writer2.close()
writer3.close()
writer4.close()
writer5.close()

# plot final concentration profiles along x - line plot, since this is a 1D mesh
import matplotlib.pyplot as plt

species = [
    ("cm", cm_pp),
    ("ct", ct_pp),
    ("O", o_pp),
    ("H2O", h2o_pp),
    ("hydride", hyd_pp),
]

for label, fn in species:
    coords = fn.function_space.tabulate_dof_coordinates()[:, 0]
    order = np.argsort(coords)
    plt.plot(coords[order], fn.x.array.real[order], marker="o", markersize=3, label=label)

plt.xlabel("x")
plt.ylabel("concentration")
plt.legend()
plt.show()
