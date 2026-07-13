# ── Step 2: Transient build-up ───────────────────────────────────────────────
import festim as F
import numpy as np
import matplotlib.pyplot as plt

kB  = 8.617e-5
R   = 0.5e-3
N   = 200
T_K = 773.0
S_T = 2e-3

li2o_t = F.Material(D_0=1.16e-5, E_D=1.047)
D_773  = li2o_t.D_0 * np.exp(-li2o_t.E_D / (kB * T_K))
tau    = R**2 / (np.pi**2 * D_773)
print(f"τ_diff = {tau:.1f} s")

r  = np.linspace(0, R, N)
dr = np.diff(r)

mesh_t    = F.Mesh1D(vertices=r, coordinate_system="spherical")
vol_t     = F.VolumeSubdomain1D(id=1, borders=[0, R], material=li2o_t)
surf_t    = F.SurfaceSubdomain1D(id=2, x=R)
tritium_t = F.Species("T")

model_t = F.HydrogenTransportProblem(
    mesh=mesh_t,
    subdomains=[vol_t, surf_t],
    species=[tritium_t],
    sources=[F.ParticleSource(value=S_T, volume=vol_t, species=tritium_t)],
    boundary_conditions=[F.DirichletBC(subdomain=surf_t, value=0.0, species=tritium_t)],
    temperature=T_K,
    settings=F.Settings(atol=1e-10, rtol=1e-10, transient=True, final_time=3 * tau),
)
model_t.settings.stepsize = F.Stepsize(tau / 100)
model_t.show_progress_bar = False   # ← must be set BEFORE initialise()
model_t.initialise()

times_out, inventory, surface_flux = [], [], []

while model_t.t.value < model_t.settings.final_time:
    model_t.iterate()

    c = tritium_t.post_processing_solution.x.array

    r_mid = 0.5 * (r[:-1] + r[1:])
    c_mid = 0.5 * (c[:-1] + c[1:])
    I = np.sum(c_mid * 4 * np.pi * r_mid**2 * dr)
    J = -D_773 * (c[-1] - c[-2]) / dr[-1] * 4 * np.pi * R**2

    times_out.append(float(model_t.t))
    inventory.append(I)
    surface_flux.append(J)

t = np.array(times_out) / tau
I = np.array(inventory)
J = np.array(surface_flux)

I_ss = S_T * 4 * np.pi * R**5 / (6 * D_773) * (2 / 15)
J_ss = S_T * (4 / 3) * np.pi * R**3
print(f"I_final / I_ss = {I[-1]/I_ss:.3f}  (expect ~1)")
print(f"J_final / J_ss = {J[-1]/J_ss:.3f}  (expect ~1)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(t, I * 1e12, color="#3881C3")
ax1.axhline(I_ss * 1e12, color="#FC9D06", ls="--", label="Analytic $I_{ss}$")
ax1.set_xlabel("t / τ"); ax1.set_ylabel("Inventory (pmol)"); ax1.set_title("Inventory"); ax1.legend()
ax2.plot(t, np.abs(J) * 1e12, color="#3881C3")
ax2.axhline(J_ss * 1e12, color="#FC9D06", ls="--", label="Analytic $J_{ss}$")
ax2.set_xlabel("t / τ"); ax2.set_ylabel("Flux (pmol/s)"); ax2.set_title("Surface flux"); ax2.legend()
fig.tight_layout(); plt.show()