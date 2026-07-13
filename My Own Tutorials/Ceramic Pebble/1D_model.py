import festim as F
import matplotlib.pyplot as plt
import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
kB = 8.617e-5   # eV/K

# Geometry
L = 1e-3 # 1 mm radius
R = L/2

N = 200 # number of points in the mesh

# ── Material properties ─────────────────────────────────────────────────────
D_0_li2O = 1.16e-5
E_D_li2O = 1.047

li2o = F.Material(D_0_li2O, E_D_li2O)

# Operating Conditions
T_K = 773 # K
S_T = 2e-3 # tritium generation rate [mol/m3/2]


# ── Diffusion coefficient at operating temperature ────────────────────────────
D_773 = li2o.D_0 * np.exp(-li2o.E_D / (kB * T_K))
print(f"D(773 K) = {D_773:.3e} m²/s")

# ── Analytical solution: c(r) = (S_T / 6D) * (R2-r2) ──────────────────────
c_max_analytical = (S_T / (6 * D_773)) * R**2
print(f"Analytical c_max at r=0: {c_max_analytical:.4e} mol/m³")

# --- Begin building cylindrical model in FESTIM ---
mesh = F.Mesh1D(vertices= np.linspace(0, R, N),
                coordinate_system = 'spherical'
)

# define subdomains
vol = F.VolumeSubdomain1D(id=1, borders = [0,R], material = li2o)
surf = F.SurfaceSubdomain1D(id=2, x = [R])

# --- Species ---
tritium = F.Species('T')

# --- Source term ---
source = F.ParticleSource(value = S_T, volume = vol, species = tritium)

# --- Bounndary Conditions ---
bc_surface = F.DirichletBC(subdomain = surf, value = 0.0, species = tritium)

# --- Model Run ---
model = F.HydrogenTransportProblem(
    mesh = mesh,
    subdomains = [vol, surf],
    species = [tritium],
    sources = [source],
    boundary_conditions = [bc_surface],
    temperature= T_K,
    settings = F.Settings(atol = 1e-10, rtol = 1e-10, transient=False)
)

model.initialise()
model.run()
print("Done.")

# ── Extract solution ──────────────────────────────────────────────────────────
c_sol    = tritium.post_processing_solution
x_coords = c_sol.function_space.mesh.geometry.x[:, 0]   # radial coordinates
c_values = c_sol.x.array[:]

sort_idx = np.argsort(x_coords)
r_sorted = x_coords[sort_idx]
c_sorted = c_values[sort_idx]

# ── Analytical profile ────────────────────────────────────────────────────────
r_plot       = np.linspace(0, R, 300)
c_analytical = (S_T / (6 * D_773)) * (R**2 - r_plot**2)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(r_sorted * 1e3, c_sorted, s=4, label="FESTIM (1D spherical)", color="steelblue")
ax.plot(r_plot * 1e3, c_analytical, "r-", lw=2, label="Analytical")
ax.set(xlabel="r [mm]", ylabel="c [mol/m³]", title="Steady-state pebble — 1D spherical")
ax.legend()
plt.tight_layout()
plt.show()

# ── Numerical check ───────────────────────────────────────────────────────────
c_max_festim = np.max(c_sorted)
error = abs(c_max_festim - c_max_analytical) / c_max_analytical * 100
print(f"c_max  FESTIM:     {c_max_festim:.4e} mol/m³")
print(f"c_max  analytical: {c_max_analytical:.4e} mol/m³")
print(f"Error: {error:.3f}%")



# ------------------------------------
# -- TRANSIENT SOLUTION --
# --------------------------------

# ── Diffusion timescale ───────────────────────────────────────────────────────
tau_diff = R**2 / (np.pi**2 * D_773)
print(f"τ_diff = {tau_diff:.2e} s  ({tau_diff/3600:.2f} hours)")

# ── Analytical steady-state inventory (for normalisation) ─────────────────────
# Integrate c(r) = (S_T/6D)(R²-r²) over sphere: I_ss = 4π·S_T·R⁵ / (45·D)
I_ss_analytical = (4 * np.pi * S_T * R**5) / (45 * D_773)
print(f"Steady-state inventory (analytical): {I_ss_analytical:.4e} mol")


# --- REBUILD MODEL FOR TRANSIENT ---

mesh_t = F.Mesh1D(vertices= np.linspace(0, R, N),
                coordinate_system = 'spherical'
)
vol_t     = F.VolumeSubdomain1D(id=1, borders=[0, R], material=li2o)
surf_t    = F.SurfaceSubdomain1D(id=2, x=R)
tritium_t = F.Species("T")


flux_t = F.SurfaceFlux(surface=surf_t, field=tritium_t)
inv_t  = F.TotalVolume(field=tritium_t, volume=vol_t)
source_t = F.ParticleSource(value=S_T, volume=vol_t, species=tritium_t)
bc_surface_t = F.DirichletBC(subdomain=surf_t, value=0.0, species=tritium_t)

model_t = F.HydrogenTransportProblem(
    mesh=mesh_t,
    subdomains=[vol_t, surf_t],
    species=[tritium_t],
    sources=[source_t],
    boundary_conditions=[bc_surface_t],
    temperature=T_K,
    settings=F.Settings(
        atol=1e-10, rtol=1e-10,
        transient=True,
        final_time=3 * tau_diff,
    ),
)

model_t.settings.stepsize = F.Stepsize(
    initial_value=tau_diff * 1e-3,
    growth_factor=1.3,
    cutback_factor=0.8,
    target_nb_iterations=4,
    max_stepsize=tau_diff / 20,
)

print(f"Running to t = {3*tau_diff:.2e} s  (3 × τ_diff) ...")
model_t.initialise()
model_t.run()
print("Done.")

# ── Snapshot times ────────────────────────────────────────────────────────────
n_snaps = 8
t_snaps = np.linspace(0, 3 * tau_diff, n_snaps + 1)[1:]   # skip t=0


t_arr = np.array(inv_t.t)
I_arr = np.array(inv_t.data)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Inventory normalised to steady-state
axes[0].plot(t_arr / tau_diff, I_arr / I_ss_analytical, color="steelblue", lw=2)
axes[0].axhline(1.0, ls="--", color="red", lw=1, label="Steady state")
axes[0].axvline(1.0, ls=":", color="grey", lw=1, label="τ_diff")
axes[0].set(xlabel="t / τ_diff", ylabel="I(t) / I_ss",
            title="Inventory build-up")
axes[0].legend()

# Surface flux (what the purge gas sees)
J_arr = np.abs(np.array(flux_t.data))
axes[1].plot(t_arr / tau_diff, J_arr, color="tomato", lw=2)
axes[1].set(xlabel="t / τ_diff", ylabel="|Flux| [mol/m²/s]",
            title="Surface flux (purge gas load)")

plt.tight_layout()
plt.show()

print(f"At t = 3τ_diff, inventory is {I_arr[-1]/I_ss_analytical*100:.1f}% of steady state")