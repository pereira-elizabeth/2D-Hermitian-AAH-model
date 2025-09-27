import numpy as np
from numpy.linalg import eigh
from src.hamiltonian import aah_2d_hamiltonian  

def test_hamiltonian_is_real_symmetric_small_grid():
    # Small grid keeps the test fast
    Lx, Ly = 4, 3
    N = Lx * Ly

    # example parameters (anything reasonable is fine)
    tx, ty = 1.0, 1.0
    lambdax, alphax, phix = 0.7, (np.sqrt(5)-1)/2, 0.3
    lambday, alphay, phiy = 0.5, (np.sqrt(5)-1)/2, 0.1

    H = aah_2d_hamiltonian(
        Lx=Lx, Ly=Ly,
        tx=tx, ty=ty,
        lambdax=lambdax, alphax=alphax, phix=phix,
        lambday=lambday, alphay=alphay, phiy=phiy,
        bc="open", dtype=float
    )

    assert H.shape == (N, N)
    assert np.isrealobj(H)
    # Hermitian check (for real matrices, symmetric)
    assert np.allclose(H, H.T, atol=1e-12)

def test_eigen_reconstruction_matches():
    Lx, Ly = 3, 3
    N = Lx * Ly
    tx, ty = 1.0, 0.8
    lambdax, alphax, phix = 0.4, 0.377, 0.0
    lambday, alphay, phiy = 0.6, 0.289, 0.2

    H = aah_2d_hamiltonian(
        Lx=Lx, Ly=Ly,
        tx=tx, ty=ty,
        lambdax=lambdax, alphax=alphax, phix=phix,
        lambday=lambday, alphay=alphay, phiy=phiy,
        bc="open", dtype=float
    )

    w, V = eigh(H)               # V columns are eigenvectors
    H_rec = (V * w) @ V.T        # V diag(w) V^T
    assert np.allclose(H, H_rec, atol=1e-8)

def test_phi_x_sweep_shapes():
    Lx, Ly = 3, 2
    N = Lx * Ly
    tx = ty = 1.0
    lambdax, alphax = 0.5, 0.377
    lambday, alphay, phiy = 0.0, 0.0, 0.0   # simple y-channel

    phix_vals = np.linspace(0.0, 2*np.pi, 3, endpoint=False)
    spectra = []
    for phix in phix_vals:
        H = aah_2d_hamiltonian(
            Lx=Lx, Ly=Ly,
            tx=tx, ty=ty,
            lambdax=lambdax, alphax=alphax, phix=phix,
            lambday=lambday, alphay=alphay, phiy=phiy,
            bc="open", dtype=float
        )
        w = eigh(H, UPLO='U')[0]
        spectra.append(w)

    E = np.stack(spectra, axis=0)  # (n_phi, N)
    assert E.shape == (len(phix_vals), N)
    assert np.all(np.isfinite(E))
