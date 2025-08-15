"""D2Q9 velocity set.

Defines the nine discrete velocities for the D2Q9 lattice Boltzmann
scheme. The directions are ordered as:

0. rest
1. east
2. north
3. west
4. south
5. northeast
6. northwest
7. southwest
8. southeast

Velocities are expressed in lattice units with lattice spacing (Δx) and
time step (Δt) equal to one.

Attributes
----------
E : ndarray of shape (9, 2)
    Velocity vectors ``[e_x, e_y]`` following the standard ordering.
E_LENGTHS : ndarray of shape (9,)
    Euclidean length of each velocity vector, useful for Bouzidi
    normalization.
"""

import numpy as np

# D2Q9 discrete velocity set in (ex, ey) format
# Standard ordering:
# 0: rest, 1: east, 2: north, 3: west, 4: south,
# 5: northeast, 6: northwest, 7: southwest, 8: southeast
E = np.array(
    [
        [0, 0],  # e0
        [1, 0],  # e1
        [0, 1],  # e2
        [-1, 0],  # e3
        [0, -1],  # e4
        [1, 1],  # e5
        [-1, 1],  # e6
        [-1, -1],  # e7
        [1, -1],  # e8
    ],
    dtype=int,
)

# Physical lengths for each direction (multiples of lattice spacing Δx)
# Useful for Bouzidi normalization
E_LENGTHS = np.sqrt(np.sum(E.astype(float) ** 2, axis=1))

__all__ = ["E", "E_LENGTHS"]
