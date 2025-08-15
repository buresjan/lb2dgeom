import numpy as np

# D2Q9 discrete velocity set in (ex, ey) format
# Standard ordering:
# 0: rest, 1: east, 2: north, 3: west, 4: south,
# 5: northeast, 6: northwest, 7: southwest, 8: southeast
E = np.array([
    [ 0,  0],  # e0
    [ 1,  0],  # e1
    [ 0,  1],  # e2
    [-1,  0],  # e3
    [ 0, -1],  # e4
    [ 1,  1],  # e5
    [-1,  1],  # e6
    [-1, -1],  # e7
    [ 1, -1],  # e8
], dtype=int)

# Physical lengths for each direction (multiples of lattice spacing Δx)
# Useful for Bouzidi normalization
E_LENGTHS = np.sqrt(np.sum(E.astype(float)**2, axis=1))

__all__ = ["E", "E_LENGTHS"]
