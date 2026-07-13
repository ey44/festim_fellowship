# ── Step 4 (revised): Steady-state sweep → response surface ──────────────────
import festim as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

kB   = 8.617e-5
R    = 0.5e-3
N    = 200
S_T  = 2e-3
D_0, E_D = 1.16e-5, 1.047
n_t  = 1e25
k_0, E_k = 1.2e-17, 0.2
p_0, E_p = 1e13,    1.2

T_range = np.arange(500, 1050, 50)   # 500 → 1000 K in 50 K steps

r     = np.linspace(0, R, N)
dr    = np.diff(r)
r_mid = 0.5 * (r[:-1] + r[1:])
w     = 4 * np.pi * r_mid**2 * dr
V_peb = (4/3) * np.pi * R**3
J_exact = S_T * V_peb    # exact by mass balance -- temperature-independent

# ── Sweep ─────────────────────────────────────────────────────────────────────
rows = []

for T_K in T_range:
    D = D_0 * np.exp(-E_D / (kB * T_K))
    k = k_0 * np.exp(-E_k / (kB * T_K))
    p = p_0 * np.exp(-E_p / (kB * T_K))

    li2o  = F.Material(D_0=D_0, E_D=E_D)
    mesh  = F.Mesh1D(vertices=r, coordinate_system="spherical")
    vol   = F.VolumeSubdomain1D(id=1, borders=[0, R], material=li2o)
    surf  = F.SurfaceSubdomain1D(id=2, x=R)
    T_m   = F.Species("T_m")
    T_t   = F.Species("T_t", mobile=False)
    empty = F.ImplicitSpecies(n=n_t, others=[T_t])

    model = F.HydrogenTransportProblem(
        mesh=mesh,
        subdomains=[vol, surf],
        species=[T_m, T_t],
        reactions=[F.Reaction(
            reactant=[T_m, empty], product=[T_t],
            k_0=k_0, E_k=E_k, p_0=p_0, E_p=E_p, volume=vol,
        )],
        sources=[F.ParticleSource(value=S_T, volume=vol, species=T_m)],
        boundary_conditions=[F.DirichletBC(subdomain=surf, value=0.0, species=T_m)],
        temperature=float(T_K),
        settings=F.Settings(atol=1e-10, rtol=1e-10, transient=False),
    )
    model.initialise()
    model.run()

    c_m = T_m.post_processing_solution.x.array
    c_t = T_t.post_processing_solution.x.array

    I_mob  = np.sum(0.5*(c_m[:-1]+c_m[1:]) * w)
    I_trap = np.sum(0.5*(c_t[:-1]+c_t[1:]) * w)
    J_sim  = abs(-D * (c_m[-1] - c_m[-2]) / dr[-1] * 4*np.pi*R**2)

    # Effective equilibration timescale (analytical, dilute-trap limit)
    tau_diff = R**2 / (np.pi**2 * D)
    tau_eff  = tau_diff * (1 + k * n_t / p)

    rows.append([T_K, I_mob, I_trap, I_mob+I_trap, J_sim, tau_diff, tau_eff])
    print(f"T={T_K:4.0f}K  I_mob={I_mob:.2e}  I_trap={I_trap:.2e}  "
          f"J/J_exact={J_sim/J_exact:.3f}  τ_eff={tau_eff:.2e}s")

rows     = np.array(rows)
T_arr    = rows[:, 0]
I_mob    = rows[:, 1]
I_trap   = rows[:, 2]
I_tot    = rows[:, 3]
J_sim    = rows[:, 4]
tau_diff = rows[:, 5]
tau_eff  = rows[:, 6]
inv_T    = 1.0 / T_arr

# ── Arrhenius fits ─────────────────────────────────────────────────────────────
def arrhenius(inv_T, log_A, E_eff):
    return log_A - E_eff * inv_T / kB

# Fit log(I_mob), log(I_trap), log(tau_eff) vs 1/T
pI_mob,  _ = curve_fit(arrhenius, inv_T, np.log(I_mob))
pI_trap, _ = curve_fit(arrhenius, inv_T, np.log(I_trap))
pTau,    _ = curve_fit(arrhenius, inv_T, np.log(tau_eff))

T_fine   = np.linspace(480, 1020, 200)
inv_fine = 1 / T_fine

print(f"\nFit results:")
print(f"  I_mob:   A={np.exp(pI_mob[0]):.3e}, E_eff={pI_mob[1]:.3f} eV")
print(f"  I_trap:  A={np.exp(pI_trap[0]):.3e}, E_eff={pI_trap[1]:.3f} eV")
print(f"  τ_eff:   A={np.exp(pTau[0]):.3e}, E_eff={pTau[1]:.3f} eV")

# ── Reduced-order model functions (for later integration) ─────────────────────
def I_mob_ss(T):
    return np.exp(arrhenius(1/T, *pI_mob))

def I_trap_ss(T):
    return np.exp(arrhenius(1/T, *pI_trap))

def tau_eff_fn(T):
    return np.exp(arrhenius(1/T, *pTau))

def I_tot_ss(T):
    return I_mob_ss(T) + I_trap_ss(T)

# ── Quick demo: reduced-order ODE for one pebble journey ──────────────────────
from scipy.integrate import solve_ivp

t_journey = 10 * (R**2 / (np.pi**2 * D_0 * np.exp(-E_D / (kB * 773.0))))
t_ramp    = 0.2 * t_journey
T_min, T_max_j = 600.0, 900.0

def T_profile(t):
    if t < t_ramp:
        return T_min + (T_max_j - T_min) * t / t_ramp
    elif t < t_journey - t_ramp:
        return T_max_j
    else:
        return T_min + (T_max_j - T_min) * (t_journey - t) / t_ramp

def dI_dt(t, I):
    T = T_profile(t)
    return (S_T * V_peb - (I[0] - I_tot_ss(T)) / tau_eff_fn(T),)

t_eval = np.linspace(0, t_journey, 500)
sol = solve_ivp(dI_dt, [0, t_journey], [0.0], t_eval=t_eval, method="RK45")
T_traj = np.array([T_profile(t) for t in t_eval])

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. Arrhenius plot
ax = axes[0, 0]
ax.semilogy(1000/T_arr, I_mob  * 1e12, 'o', color="#3881C3", label="I_mob (FESTIM)")
ax.semilogy(1000/T_arr, I_trap * 1e12, 's', color="#FC9D06", label="I_trap (FESTIM)")
ax.semilogy(1000/T_fine, np.exp(arrhenius(inv_fine, *pI_mob))  * 1e12, '-', color="#3881C3", alpha=0.6)
ax.semilogy(1000/T_fine, np.exp(arrhenius(inv_fine, *pI_trap)) * 1e12, '-', color="#FC9D06", alpha=0.6)
ax.set_xlabel("1000/T (K⁻¹)"); ax.set_ylabel("Inventory (pmol)")
ax.set_title("Arrhenius plot -- steady-state inventories"); ax.legend(fontsize=8)

# 2. τ_eff vs T
ax = axes[0, 1]
ax.semilogy(T_arr, tau_diff / 3600, 'o', color="#3881C3", label="τ_diff")
ax.semilogy(T_arr, tau_eff  / 3600, 's', color="#FC9D06", label="τ_eff (with traps)")
ax.semilogy(T_fine, np.exp(arrhenius(inv_fine, *pTau)) / 3600, '-', color="#FC9D06", alpha=0.6)
ax.set_xlabel("T (K)"); ax.set_ylabel("Timescale (hours)")
ax.set_title("Equilibration timescales"); ax.legend(fontsize=8)

# 3. J_ss verification
ax = axes[1, 0]
ax.plot(T_arr, J_sim / J_exact, 'o', color="#3881C3")
ax.axhline(1.0, color="#FC9D06", ls="--")
ax.set_xlabel("T (K)"); ax.set_ylabel("J_sim / J_exact")
ax.set_title("Flux mass-balance check (should be 1.0 everywhere)")
ax.set_ylim(0.9, 1.1)

# 4. Reduced-order ODE journey
ax = axes[1, 1]
ax2 = ax.twinx()
ax.plot(t_eval/3600, sol.y[0] * 1e12, color="#3881C3", lw=2, label="I_total (ODE)")
ax.plot(t_eval/3600, I_tot_ss(T_traj) * 1e12, color="#3881C3", ls=":", lw=1, label="I_ss(T(t))")
ax2.plot(t_eval/3600, T_traj, color="#FC9D06", lw=1.5, alpha=0.7)
ax2.set_ylabel("T (K)", color="#FC9D06")
ax.set_xlabel("Journey time (hours)"); ax.set_ylabel("Inventory (pmol)")
ax.set_title("Reduced-order ODE: one pebble journey"); ax.legend(fontsize=8)

fig.tight_layout()
plt.savefig("response_surface.png", dpi=150)
plt.show()