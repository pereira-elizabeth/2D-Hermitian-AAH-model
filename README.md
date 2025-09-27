# 2D Hermitian Aubry–André–Harper (AAH) Model

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Last Commit](https://img.shields.io/github/last-commit/pereira-elizabeth/aah-2d-hermitian)
![Code Size](https://img.shields.io/github/languages/code-size/pereira-elizabeth/aah-2d-hermitian)
<!-- CI badge (works after you add .github/workflows/tests.yml) -->
<!-- ![Build](https://github.com/pereira-elizabeth/aah-2d-hermitian/actions/workflows/tests.yml/badge.svg) -->
A simple simulation of how electrons or light behave in a 2D crystal with a patterned potential.

This repository contains a **minimal, safe-to-share implementation** of the **2D Hermitian Aubry–André–Harper model**.  
It builds the tight-binding Hamiltonian on an $L_x \times L_y$ lattice, diagonalizes it, and plots the energy spectrum $E$ as a function of the modulation phase $\phi_x$.

---

## What is implemented?

- A real-symmetric Hamiltonian  
  $H \in \mathbb{R}^{(L_x L_y) \times (L_x L_y)}$  

- Onsite AAH potential that depends on both lattice directions $x$ and $y$.

- Nearest-neighbor hopping on a 2D square grid  
- Open boundary conditions (easy to modify if you want periodic)  
- Fast diagonalization using `numpy.linalg.eigh`  
- Simple plotting routine: spectrum $E$ vs $\phi_x$

---

## Example Output

Running the demo script will generate a plot of the energy bands vs $\phi_x$, e.g.:
![Energy spectra plot](figures/spectrum.png)

---

## How to run

Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/2D-Hermitian-AAH-model.git
cd 2D-Hermitian-AAH-model
pip install -r requirements.txt


