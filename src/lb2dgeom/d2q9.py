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
    Euclidean length of each velocity vector, useful for Bouzidi normalization.
E_NAMES : list[str]
    Direction names corresponding to indices ``0..8`` in ``E``.
    The mapping is::

        0: rest
        1: east        (1, 0)
        2: north       (0, 1)
        3: west       (-1, 0)
        4: south       (0,-1)
        5: northeast   (1, 1)
        6: northwest  (-1, 1)
        7: southwest  (-1,-1)
        8: southeast   (1,-1)
 
    When exporting as text via :func:`lb2dgeom.io.save_txt`, columns are
    written in the order ``q_east, q_north, q_west, q_south, q_northeast,
    q_northwest, q_southwest, q_southeast`` corresponding to indices 1..8.
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

E_NAMES = [
    "rest",
    "east",
    "north",
    "west",
    "south",
    "northeast",
    "northwest",
    "southwest",
    "southeast",
]

__all__ = ["E", "E_LENGTHS", "E_NAMES"]
