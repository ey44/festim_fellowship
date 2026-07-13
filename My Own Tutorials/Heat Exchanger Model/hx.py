import numpy as np
import festim as F

T = 580 + 273 # K
p_T2 = 10 # Pa
hx_wall_thickness = 2.5e-3 # mm

n_mesh = 201
mesh = F.Mesh1D(np.linspace(0, hx_wall_thickness, n_mesh))

from material_library import get_material, M_T, M_H

mat = get_material('Inconel617')
KJMOL_TO_EV = 1.0 / 96.485

D_0_T = mat['D0'] / np.sqrt(M_T / M_H)
E_D   = mat['ED'] * KJMOL_TO_EV

inconel = F.Material(D_0=D_0_T, E_D=E_D)

vol = F.VolumeSubdomain1D(id = 1, borders = [0, hx_wall_thickness],
                           material = inconel)

he_side = F.SurfaceSubdomain1D(id=2, x=0.0) # helium side is at 0
steam_side = F.SurfaceSubdomain1D(id=3, x=hx_wall_thickness) # steam side is at the wall thickness

tritium = F.Species('T') # we are doing tritium

S_0 = 2.0 * mat['KS0']             # atomic, S_0 = 2*K_S
E_S = mat['ES'] * KJMOL_TO_EV


bcs = [
    F.SievertsBC(subdomain=he_side, S_0=S_0, E_S=E_S, pressure=p_T2, species=tritium), # setting the sieverts condition on one side
    F.FixedConcentrationBC(subdomain=steam_side, value=0.0, species=tritium), # setting the fixed concentration condition on the other side, assuming a perfect sink
]

flux_out = F.SurfaceFlux(field=tritium, surface=steam_side) # we are interested in the flux out of here


model = F.HydrogenTransportProblem(
    mesh=mesh,
    subdomains=[vol, he_side, steam_side],
    species=[tritium],
    boundary_conditions=bcs,
    exports=[flux_out],
    temperature = T,
    settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
)

model.initialise()
model.run()

J_atomic = abs(flux_out.data[-1])
J_T2 = J_atomic / 2.0
print(f"J_T2 = {J_T2:.4e} mol/m2/s")
print(flux_out.data)
