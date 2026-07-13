# ── Step 3: Transient with one McNabb-Foster trap ────────────────────────────
import festim as F
import numpy as np
import matplotlib.pyplot as plt

kB  = 8.617e-5
R   = 0.5e-3
N   = 200
T_K = 773.0
S_T = 2e-3

li2o = F.Material(D_0=1.16e-5, E_D=1.047)
D    = li2o.D_0 * np.exp(-li2o.E_D / (kB * T_K))
tau  = R**2 / (np.pi**2 * D)
print(f"D = {D:.3e} m²/s,  τ_diff = {tau:.1f} s")

# ── Trap parameters (representative for Li₂O ceramic) ────────────────────────
n_t = 1e25          # trap density, m⁻³
k_0 = 1.2e-17       # trapping pre-exp, m³/s
E_k = 0.2           # trapping activation energy, eV  (low → fast trapping)
p_0 = 1e13          # detrapping pre-exp, s⁻¹  (attempt frequency)
E_p = 1.2           # detrapping activation energy, eV  (deeper than E_D)

k = k_0 * np.exp(-E_k / (kB * T_K))   # trapping rate at 773 K
p = p_0 * np.exp(-E_p / (kB * T_K))   # detrapping rate at 773 K
print(f"k(773K) = {k:.3e} m³/s,  p(773K) = {p:.3e} s⁻¹")
print(f"Trap occupancy at SS ~ {k * (S_T/6/D) * R**2 / (k * (S_T/6/D) * R**2 + p):.2%}")

# ── Mesh ──────────────────────────────────────────────────────────────────────
r  = np.linspace(0, R, N)
dr = np.diff(r)

mesh = F.Mesh1D(vertices=r, coordinate_system="spherical")
vol  = F.VolumeSubdomain1D(id=1, borders=[0, R], material=li2o)
surf = F.SurfaceSubdomain1D(id=2, x=R)

# ── Species ───────────────────────────────────────────────────────────────────
T_m = F.Species("T_m")                    # mobile
T_t = F.Species("T_t", mobile=False)      # trapped (no diffusion term)
empty = F.ImplicitSpecies(n=n_t, others=[T_t])  # empty sites = n_t - c_trapped

# ── Reaction: T_m + empty ⇌ T_t ──────────────────────────────────────────────
reaction = F.Reaction(
    reactant=[T_m, empty],
    product=[T_t],
    k_0=k_0, E_k=E_k,
    p_0=p_0, E_p=E_p,
    volume=vol,
)

# ── Model ─────────────────────────────────────────────────────────────────────
model = F.HydrogenTransportProblem(
    mesh=mesh,
    subdomains=[vol, surf],
    species=[T_m, T_t],
    reactions=[reaction],
    sources=[F.ParticleSource(value=S_T, volume=vol, species=T_m)],
    boundary_conditions=[F.DirichletBC(subdomain=surf, value=0.0, species=T_m)],
    temperature=T_K,
    settings=F.Settings(atol=1e-10, rtol=1e-10, transient=True, final_time=5 * tau),
)
model.settings.stepsize = F.Stepsize(tau / 100)
model.show_progress_bar = False
model.initialise()

# ── Time loop ─────────────────────────────────────────────────────────────────
times, I_mob, I_trap, J_surf = [], [], [], []

while model.t.value < model.settings.final_time:
    model.iterate()

    c_m = T_m.post_processing_solution.x.array
    c_t = T_t.post_processing_solution.x.array

    r_mid = 0.5 * (r[:-1] + r[1:])
    w     = 4 * np.pi * r_mid**2 * dr          # shell volumes

    times.append(float(model.t))
    I_mob.append(np.sum(0.5*(c_m[:-1]+c_m[1:]) * w))
    I_trap.append(np.sum(0.5*(c_t[:-1]+c_t[1:]) * w))
    J_surf.append(-D * (c_m[-1] - c_m[-2]) / dr[-1] * 4*np.pi*R**2)

t      = np.array(times) / tau
I_mob  = np.array(I_mob)
I_trap = np.array(I_trap)
J_surf = np.array(J_surf)
I_tot  = I_mob + I_trap

# Trap-free steady-state references
I_ss = S_T * 4*np.pi*R**5 / (6*D) * (2/15)
J_ss = S_T * (4/3)*np.pi*R**3
print(f"\nAt t=5τ:  I_mob/I_ss = {I_mob[-1]/I_ss:.3f}")
print(f"          I_trap     = {I_trap[-1]:.3e} mol")
print(f"          J/J_ss     = {J_surf[-1]/J_ss:.3f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(t, I_mob  * 1e12, color="#3881C3", label="Mobile")
ax1.plot(t, I_trap * 1e12, color="#FC9D06", label="Trapped")
ax1.plot(t, I_tot  * 1e12, color="#26252C", ls="--", label="Total")
ax1.axhline(I_ss   * 1e12, color="gray",    ls=":",  label="No-trap $I_{ss}$")
ax1.set_xlabel("t / τ"); ax1.set_ylabel("Inventory (pmol)")
ax1.set_title("Inventory with trap"); ax1.legend(fontsize=8)

ax2.plot(t, np.abs(J_surf) * 1e12, color="#3881C3", label="With trap")
ax2.axhline(J_ss * 1e12, color="gray", ls=":", label="No-trap $J_{ss}$")
ax2.set_xlabel("t / τ"); ax2.set_ylabel("Release flux (pmol/s)")
ax2.set_title("Surface flux with trap"); ax2.legend(fontsize=8)

fig.tight_layout(); plt.show()