# Example 3: Two species in 1D

## Problem statement

Now we solve for **two fields at once** on the same rod `Ω = (0, 1)`. Think of two
chemical species, `c1` and `c2`, each diffusing independently with their own boundary
conditions. Both start at zero and are driven in from the left.

- The domain is the 1D interval `Ω = (0, 1)`.
- Both species share a diffusion coefficient `k = 0.1`.
- Initial condition: `c1 = c2 = 0` everywhere.
- Species `c1`: left end `= 100`, right end `= 0`.
- Species `c2`: left end `= 75`, right end `= 0`.
- Time step `dt = 0.1`, final time `t = 10`.

The two species do **not** interact here; they are solved together purely to introduce
the machinery for multiple fields. Coupling them is the subject of Example 4.

### Strong form

For each species `i` in `{1, 2}`:

$$
\frac{\partial c_i}{\partial t} - \nabla \cdot (k \, \nabla c_i) = 0
\quad \text{in } \Omega, \quad t > 0,
$$

with the Dirichlet values listed above and `c_i(x, 0) = 0`.

## What this example teaches (new compared to Example 2)

1. **A mixed function space.** `basix.ufl.mixed_element([...])` packs two fields into a
   single function `u`, solved together in one system.
2. **Splitting and collapsing.** `ufl.split(u)` gives the symbolic fields `c1, c2` for
   the weak form; `u.sub(i).collapse()` extracts a standalone function for output.
3. **Sub-space boundary conditions.** Conditions are applied to `V.sub(0)` and
   `V.sub(1)` separately, using `locate_entities_boundary` and `meshtags` to label the
   ends.
4. **One weak form, several equations**, summed into a single residual `F`.
5. **Writing one `.bp` file per species** for visualisation.

## Files

- [`my_script.py`](my_script.py): where **you** write the script from scratch, starting
  from the problem statement above. This is the main task.
- [`eg_3.py`](eg_3.py): a complete worked example to check your approach against. It
  writes `two_species_c1.bp` and `two_species_c2.bp`, which you can open in
  [ParaView](https://www.paraview.org/).
- [`eg_3.ipynb`](eg_3.ipynb): a step-by-step walkthrough of the worked example, there
  to help if you get stuck.

See the [top-level README](../README.md) for how to build the environment and the
intended workflow.
