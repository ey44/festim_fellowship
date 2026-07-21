# Example 2: Transient heat transfer in 1D

## Problem statement

This is Example 1 with **time** added. The same rod `Ω = (0, 1)` starts cold
(`T = 0` everywhere) and we watch it heat up as the hot left end drives heat in.

- The domain is the 1D interval `Ω = (0, 1)`.
- Constant thermal conductivity `k = 0.1`.
- Initial condition: `T(x, 0) = 0`.
- The left end is held at `T = 100`, the right end at `T = 0` for all time.
- Time step `dt = 0.1`, final time `t = 10`.

Find how the temperature `T(x, t)` evolves in time. As `t` grows the solution
relaxes towards the straight-line steady state found in Example 1.

### Strong form

Transient heat conduction with no source:

$$
\frac{\partial T}{\partial t} - \nabla \cdot (k \, \nabla T) = 0
\quad \text{in } \Omega, \quad t > 0,
$$

with `T(0, t) = 100`, `T(1, t) = 0` and `T(x, 0) = 0`.

### Time discretisation

We march in time with a **backward Euler** scheme: the time derivative becomes
`(T - T_n) / dt`, where `T_n` is the solution at the previous step. At each step we
solve a problem that looks just like Example 1, plus this extra term.

## What this example teaches (new compared to Example 1)

1. **A second function `u_n`** holding the solution at the previous time step.
2. **The time-derivative term** `((u - u_n) / dt) * v * dx` added to the weak form.
3. **A time loop:** solve, copy `u` into `u_n`, advance `t`, repeat.
4. **Writing results to file** with `VTXWriter` (a `.bp` folder you open in ParaView)
   instead of plotting a single static figure.
5. **A `tqdm` progress bar** to track the time stepping.

## Files

- [`my_script.py`](my_script.py): where **you** write the script from scratch, starting
  from the problem statement above. This is the main task.
- [`eg_2.py`](eg_2.py): a complete worked example to check your approach against. It
  writes `ht_transient.bp`, which you can open in [ParaView](https://www.paraview.org/)
  to scrub through time.
- [`eg_2.ipynb`](eg_2.ipynb): a step-by-step walkthrough of the worked example, there
  to help if you get stuck.

See the [top-level README](../README.md) for how to build the environment and the
intended workflow.
