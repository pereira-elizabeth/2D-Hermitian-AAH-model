# examples/aah_2d_demo.py
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

def main():
    os.makedirs("figures", exist_ok=True)

    # Lattice & model params (match your original choices)
    Lx = 8
    Ly = 8
    t = 1.0
    lamx = 3.5
    lamy = 3.5
    alphax = 1.0 / 8.0
    alphay = 1.0 / 8.0
    phiy = -np.pi * (Lx + 1)  # you can hold φ_y fixed if you like
    periodic_x = False
    periodic_y = False

    # Scan φ_x from 0 to 2π
    phix_vals = np.linspace(0.0, 2.0 * np.pi, 301)
    eigs_list = []

    for phix in phix_vals:
        H = aah_2d_hamiltonian(
            Lx=Lx, Ly=Ly, t=t,
            lamx=lamx, lamy=lamy,
            alphax=alphax, alphay=alphay,
            phix=phix, phiy=phiy,
            periodic_x=periodic_x, periodic_y=periodic_y,
        )
        # Hermitian => use eigh (real, sorted)
        w = np.linalg.eigh(H)[0]
        eigs_list.append(w)

    eigs = np.array(eigs_list)  # shape: [len(phix_vals), Lx*Ly]

    # Plot spectrum E vs φ_x / (2π)
    xaxis = phix_vals / (2.0 * np.pi)
    plt.figure(figsize=(7, 4))
    for n in range(eigs.shape[1]):
        plt.plot(xaxis, eigs[:, n], ".", markersize=1, color= 'black')
    plt.xlabel(r"$\phi_x / 2\pi$")
    plt.ylabel(r"$E$")
    bc = "open"
    if periodic_x or periodic_y:
        bc = f"periodic in {'x' if periodic_x else ''}{' & ' if (periodic_x and periodic_y) else ''}{'y' if periodic_y else ''}"
    plt.title(f"2D AAH, L={Lx}×{Ly}, t={t}, λx=λy={lamx}, {bc}")
    plt.tight_layout()
    out = "figures/spectrum.png"
    plt.savefig(out, dpi=160)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()

