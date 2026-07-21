# Example 1: Steady state heat transfer in 1D

## Problem statement

We want to compute the steady-state temperature profile in a thin rod of length
`L = 1` (dimensionless units).

- The domain is the 1D interval `Ω = (0, 1)`.
- The material has a constant thermal conductivity `k = 0.1`.
- There is **no heat source** inside the rod.
- The left end is held at a fixed temperature `T = 100`.
- The right end is held at a fixed temperature `T = 0`.

Find the temperature `T(x)` everywhere in the rod.

### Strong form

Steady-state heat conduction with no source is governed by:

$$
-\nabla \cdot (k \, \nabla T) = 0 \quad \text{in } \Omega = (0, 1)
$$

with **Dirichlet** (fixed-value) boundary conditions:

$$
T(0) = 100, \qquad T(1) = 0.
$$

### Analytical solution

Because `k` is constant and there is no source, the temperature varies linearly:

$$
T(x) = 100\,(1 - x).
$$

The constant heat flux is

$$
q = -k \frac{dT}{dx} = -0.1 \times (-100) = 10.
$$

This gives us a known answer to check the numerical solution against.

## What this example teaches

This is the simplest possible finite-element problem, so it's a good place to see
the **anatomy of a DOLFINx script**. Every simulation in this tutorial is built
from the same building blocks:

1. **Backends:** `MPI` (parallelism) and `PETSc` (linear algebra, solvers).
2. **Mesh:** the discretised geometry of the domain `Ω`.
3. **Function space:** the space of allowed solutions (here, piecewise-linear "hat"
   functions), plus the **unknown** function `u` and a **test** function `v`.
4. **Boundary conditions:** pinning the solution at the two ends.
5. **Variational (weak) form:** the PDE rewritten as an integral equation `F = 0`.
6. **Solver:** the PETSc options and the problem object that drives Newton's method.
7. **Solve and post-process:** run it, then visualise and extract quantities.

## Files

- [`my_script.py`](my_script.py): where **you** write the script from scratch, starting
  from the problem statement above. This is the main task.
- [`eg_1.py`](eg_1.py): a complete worked example to check your approach against. The
  PyVista plotting helper is defined at the top of this file.
- [`eg_1.ipynb`](eg_1.ipynb): a step-by-step walkthrough of the worked example, there
  to help if you get stuck.

See the [top-level README](../README.md) for how to build the environment and the
intended workflow.
