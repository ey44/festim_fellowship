import festim as F
import numpy as np
import matplotlib.pyplot as plt

# same reaction network as eg_4.py, ported onto festim.HydrogenTransportProblem's
# Species/Reaction API instead of building the UFL forms (F +=, ufl.split, mixed_element...) by hand.
#
# NOT ported: eg_4.py's D_eff, cm's diffusivity smoothly interpolating between D_pristine
# and D_hydride depending on c_hyd. festim.Material.D_0/E_D accept a float/int/dict/fem.Function
# per species, not a UFL expression of another species' concentration, so there's no direct
# equivalent here. cm instead just uses the constant D_pristine below (same as if c_hyd were
# always 0). If that coupling matters, it would need a custom D_0 built as a dolfinx fem.Function
# updated by hand each step, or overriding HydrogenTransportProblem's diffusion term - out of
# scope for a straight reaction-network port.

my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh1D(np.linspace(0, 1, 100))

left_surf = F.SurfaceSubdomain1D(id=1, x=0)
right_surf = F.SurfaceSubdomain1D(id=2, x=1)

# rate constants and diffusivities, same values as eg_4.py
D_pristine = 0.1  # cm's diffusivity (D_eff's nonlinear c_hyd-dependence not ported, see note above)
D_O = 0.01  # oxygen's own diffusivity
k = 1.0  # trapping forward rate
p = 0.001  # trapping backward (detrapping) rate
n = 1  # number of trap sites
k_HO = 1e-4  # oxidation forward rate (cm + O -> H2O), no reverse in eg_4.py
k_hyd = 0.05  # hydride formation forward rate
p_hyd = 1e-6  # hydride dissociation rate, near-irreversible like eg_4.py

# cm and c_o need different diffusivities -> D_0/E_D as dicts keyed by species
# (immobile species - ct, c_h2o, c_hyd - don't need an entry, festim only looks
# up D for species with mobile=True)
material = F.Material(
    D_0={"cm": D_pristine, "O": D_O},
    E_D={"cm": 0, "O": 0},
)
vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=material)

my_model.subdomains = [vol, left_surf, right_surf]

# species: cm/c_o mobile (diffusing), ct/c_h2o/c_hyd immobile (reaction products/traps,
# same as ct/c_h2o/c_hyd being DG with no diffusion term in eg_4.py's F)
cm = F.Species("cm")
ct = F.Species("ct", mobile=False)
c_o = F.Species("O")
c_h2o = F.Species("H2O", mobile=False)
c_hyd = F.Species("hydride", mobile=False)

# empty trap sites as an implicit species: n - ct, same as eg_4.py's (n - ct) term
# written directly into `reaction`
empty_traps = F.ImplicitSpecies(n=n, others=[ct])

my_model.species = [cm, ct, c_o, c_h2o, c_hyd]

my_model.reactions = [
    # trapping: cm + empty_traps <--> ct, same as eg_4.py's `reaction`
    F.Reaction(
        reactant=[cm, empty_traps],
        product=[ct],
        k_0=k,
        E_k=0,
        p_0=p,
        E_p=0,
        volume=vol,
    ),
    # oxidation: cm + O -> H2O, one-way (no p_0), same as eg_4.py's `reaction_HO`
    F.Reaction(
        reactant=[cm, c_o],
        product=[c_h2o],
        k_0=k_HO,
        E_k=0,
        p_0=0,
        E_p=0,
        volume=vol,
    ),
    # hydride formation: cm <--> hydride, near-irreversible, same as eg_4.py's `reaction_hyd`
    F.Reaction(
        reactant=[cm],
        product=[c_hyd],
        k_0=k_hyd,
        E_k=0,
        p_0=p_hyd,
        E_p=0,
        volume=vol,
    ),
]

# only cm and O have BCs in eg_4.py - ct/c_h2o/c_hyd are purely reaction-driven
my_model.boundary_conditions = [
    F.FixedConcentrationBC(left_surf, value=100, species=cm),
    F.FixedConcentrationBC(right_surf, value=0, species=cm),
    F.FixedConcentrationBC(left_surf, value=10.0, species=c_o),
    F.FixedConcentrationBC(right_surf, value=0.0, species=c_o),
]

my_model.temperature = 300  # required attribute, D isn't actually temperature-dependent here (E_D=0)

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=10)  # same final_time as eg_4.py
my_model.settings.stepsize = F.Stepsize(0.05)  # same dt as eg_4.py

my_model.exports = [
    F.VTXSpeciesExport("trapping_cm_festim.bp", field=cm),
    F.VTXSpeciesExport("trapping_ct_festim.bp", field=ct),
    F.VTXSpeciesExport("trapping_o_festim.bp", field=c_o),
    F.VTXSpeciesExport("trapping_h2o_festim.bp", field=c_h2o),
    F.VTXSpeciesExport("trapping_hyd_festim.bp", field=c_hyd),
]

my_model.initialise()
my_model.run()


def plot_profile(species, **kwargs):
    index = my_model.species.index(species)
    V0, dofs = my_model.function_space.sub(index).collapse()
    coords = V0.tabulate_dof_coordinates()[:, 0]
    sort_coords = np.argsort(coords)
    c = my_model.u.x.array[dofs][sort_coords]
    x = coords[sort_coords]
    return plt.plot(x, c, marker="o", markersize=3, **kwargs)


for species in my_model.species:
    plot_profile(species, label=species.name)

plt.xlabel("x")
plt.ylabel("concentration")
plt.legend()
plt.show()
