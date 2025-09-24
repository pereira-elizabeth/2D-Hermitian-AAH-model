# 2D-Hermitian-AAH-model
2D Hermitian Aubry–André–Harper model: build $H(x,y)$, diagonalize, and plot $E$ vs $\phi_x$.

Minimal, **safe-to-share** implementation of the 2D Hermitian Aubry–André–Harper model on an $\(L_x \times L_y\)$ grid.
Builds the tight-binding Hamiltonian with nearest-neighbor hopping and a separable onsite modulation in $**x**$ and $**y**$, then plots the spectrum $\(E\)$ vs the phase $\(\phi_x\)$.

## Features
- Real-symmetric Hamiltonian $\(H \in \mathbb{R}^{(L_xL_y) \times (L_xL_y)}\)$
- Onsite potential depends on **both** coordinates:
 $\[
  V(x,y)=\lambda_x\sin(2\pi\alpha_x x+\phi_x)+\lambda_y\sin(2\pi\alpha_y y+\phi_y)
  \]$
- Open boundaries (simple to read/extend)
- Fast Hermitian solver (`numpy.linalg.eigh`)

## How to run
```bash
pip install -r requirements.txt
python examples/aah_2d_demo.py

