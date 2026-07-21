import festim as F
import dolfinx
from mpi4py import MPI
import numpy as np
import ufl

nx = ny = 20
mesh_fenics = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
mesh = F.Mesh(mesh=mesh_fenics)

mat = F.Material(D_0=1, E_D=0.01, thermal_conductivity=3, density=2, heat_capacity=5)

volume_subdomain = F.VolumeSubdomain(id=1, material=mat)
top_bot = F.SurfaceSubdomain(id=2, locator=lambda x: np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)))
left = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], 0.0))
right = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[0], 1.0))
subdomains = [volume_subdomain, top_bot, left, right]

heat_transfer_model = F.HeatTransferProblem()
hydrogen_problem = F.HydrogenTransportProblem()

heat_transfer_model.mesh = mesh   
hydrogen_problem.mesh = mesh

H = F.Species("H")
hydrogen_problem.species = [H]

hydrogen_problem.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=top_bot, value=1, species=H),
    F.FixedConcentrationBC(subdomain=left, value=0, species=H),
]

heat_transfer_model.subdomains = subdomains
hydrogen_problem.subdomains = subdomains

hydrogen_problem.settings = F.Settings(
    transient=True,
    atol=1e-09,
    rtol=1e-09,
    stepsize=1,
    final_time=50
)

fixed_temperature_left = F.FixedTemperatureBC(
    subdomain=left, value=lambda x: 350 + 20 * ufl.cos(x[0]) * ufl.sin(x[1])
)

heat_transfer_model.boundary_conditions = [
    fixed_temperature_left,
]

heat_transfer_model.settings = F.Settings(
    transient=True,
    atol=1e-09,
    rtol=1e-09,
    stepsize=1,
    final_time=50
)

problem = F.CoupledTransientHeatTransferHydrogenTransport(
    heat_problem=heat_transfer_model,
    hydrogen_problem=hydrogen_problem
    )
    
problem.initialise()
problem.run()

import pyvista
from dolfinx import plot  # FIXED: was missing - plot.vtk_mesh below needs this. worked in the tutorial notebook only because an earlier cell already imported it into the shared kernel namespace

pyvista.set_jupyter_backend("html")

T = problem.heat_problem.u
c = problem.hydrogen_problem.species[0].post_processing_solution
topology, cell_types, geometry = plot.vtk_mesh(T.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["T"] = T.x.array.real
u_grid.set_active_scalars("T")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, cmap="inferno", show_edges=False)
u_plotter.add_mesh(u_grid, style="wireframe", color="white", opacity=0.2)

contours = u_grid.contour(9)
u_plotter.add_mesh(contours, color="white")

u_plotter.view_xy()

u_plotter.show(screenshot="temperature.png")
u_plotter.close()

    
topology, cell_types, geometry = plot.vtk_mesh(c.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["c"] = c.x.array.real
u_grid.set_active_scalars("c")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, cmap="viridis", show_edges=False)
u_plotter.add_mesh(u_grid, style="wireframe", color="white", opacity=0.2)

contours = u_grid.contour(9)
u_plotter.add_mesh(contours, color="white")

u_plotter.view_xy()

u_plotter.show(screenshot="concentration.png")
u_plotter.close()

pyvista.close_all()
del u_plotter, u_grid, contours, topology, cell_types, geometry
import gc
gc.collect()
u_plotter.close()