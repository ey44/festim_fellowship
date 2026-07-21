# DOLFINx tutorial scripts

A set of small, self-contained finite-element examples to help new fellows get started
with **DOLFINx** and **FESTIM**. Each example introduces one new concept on top of the
previous one.

## The examples

| Folder | Topic | New concept |
| --- | --- | --- |
| [`Example_1`](Example_1) | Steady state heat transfer | The anatomy of a DOLFINx script |
| [`Example_2`](Example_2) | Transient heat transfer | Time stepping |
| [`Example_3`](Example_3) | Two species | Multiple fields (mixed function space) |
| [`Example_4`](Example_4) | Hydrogen trapping | Coupling fields through a reaction |

Work through them in order; Example 4 is the basic FESTIM model.

## Building the environment

The examples need a conda environment with DOLFINx 0.10. From this folder:

```bash
conda env create -f environment.yml
conda activate dolfinx-tutorial-env
```

(`mamba env create -f environment.yml` works too and is faster.)

## How to work through an example

Each `Example_N` folder has its own `README.md` with the full problem statement, and
the same three files:

1. **`my_script.py`**: where **you** write the script from scratch. Start from the
   problem statement and build it up yourself. This is the main task.
2. **`eg_N.py`**: a complete worked example showing one way to solve it. Use it to
   check your approach.
3. **`eg_N.ipynb`**: a step-by-step walkthrough of the worked example, there to help if
   you get stuck.

Lean on outside resources as you go, especially the official
[DOLFINx tutorial](https://jsdokken.com/dolfinx-tutorial/), which covers most of what
you need.

## Viewing results

Example 1 plots its result with PyVista. Examples 2 to 4 write `.bp` files; open these
in [ParaView](https://www.paraview.org/) to scrub through time.
