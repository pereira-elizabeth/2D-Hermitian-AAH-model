import os
import numpy as np
import matplotlib.pyplot as plt

def idx(x: int, y: int, Lx: int, Ly: int) -> int:
    """Map 2D coords (x,y) to 1D index i in row-major order."""
    return x * Ly + y

def aah_2d_hamiltonian(
    Lx: int = 8,
    Ly: int = 8,
    t: float = 1.0,
    lamx: float = 3.5,
    lamy: float = 3.5,
    alphax: float = 1.0/8.0,
    alphay: float = 1.0/8.0,
    phix: float = 0.0,
    phiy: float = 0.0,
    periodic_x: bool = False,
    periodic_y: bool = False,
) -> np.ndarray:
    """
    Build the 2D Hermitian Aubry–André–Harper Hamiltonian on an Lx × Ly lattice.

    H = sum_{x,y} [ V(x,y) |x,y><x,y| ] + t * sum_<nn> (|x,y><x',y'| + h.c.)
    with onsite modulation:
      V(x,y) = lamx * sin(2π*alphax*x + phix) + lamy * sin(2π*alphay*y + phiy)

    Returns:
        H (np.ndarray, shape [Lx*Ly, Lx*Ly], dtype=float): real symmetric Hamiltonian.
    """
    N = Lx * Ly
    H = np.zeros((N, N), dtype=float)

    # Onsite potential (depends on both x and y)
    for x in range(Lx):
        for y in range(Ly):
            i = idx(x, y, Lx, Ly)
            Vxy = (
                lamx * np.sin(2.0 * np.pi * alphax * x + phix)
                + lamy * np.sin(2.0 * np.pi * alphay * y + phiy)
            )
            H[i, i] = Vxy

    # Nearest-neighbor hopping: open BCs by default
    # Along y (horizontal neighbors within each row)
    for x in range(Lx):
        for y in range(Ly - 1):
            i = idx(x, y, Lx, Ly)
            j = idx(x, y + 1, Lx, Ly)
            H[i, j] = H[j, i] = t
        if periodic_y and Ly > 1:
            i = idx(x, Ly - 1, Lx, Ly)
            j = idx(x, 0, Lx, Ly)
            H[i, j] = H[j, i] = t

    # Along x (vertical neighbors between rows)
    for x in range(Lx - 1):
        for y in range(Ly):
            i = idx(x, y, Lx, Ly)
            j = idx(x + 1, y, Lx, Ly)
            H[i, j] = H[j, i] = t
    if periodic_x and Lx > 1:
        for y in range(Ly):
            i = idx(Lx - 1, y, Lx, Ly)
            j = idx(0, y, Lx, Ly)
            H[i, j] = H[j, i] = t

    return H

