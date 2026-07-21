from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import basix
import numpy as np
import tqdm.autonotebook
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import create_mesh, locate_entities_boundary, meshtags
import ufl

from dolfinx.io import VTXWriter
