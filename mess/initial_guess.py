"""Initial guess methods for SCF calculations."""

from typing import Tuple

import jax
import jax.numpy.linalg as jnl

from mess.types import FloatNxN


def core_hamiltonian_guess(H_core: FloatNxN, X: FloatNxN) -> FloatNxN:
    """Generate initial MO coefficients by diagonalizing core Hamiltonian.

    This produces molecular orbitals that are eigenstates of the one-electron
    Hamiltonian (kinetic + nuclear attraction), providing a reasonable starting
    point for SCF iterations.

    Args:
        H_core: Core Hamiltonian matrix (kinetic + nuclear attraction).
        X: Orthonormalization transformation matrix.

    Returns:
        MO coefficient matrix C where columns are molecular orbitals.
    """
    # Transform to orthonormal basis
    H_orth = X.T @ H_core @ X
    # Diagonalize
    _, C_orth = jnl.eigh(H_orth)
    # Transform back
    return X @ C_orth


def symmetry_broken_guess(
    H_core: FloatNxN, X: FloatNxN, n_alpha: int, n_beta: int, key: jax.Array = None
) -> Tuple[FloatNxN, FloatNxN]:
    """Generate symmetry-broken initial MO coefficients for UHF/UKS.

    Starts from core Hamiltonian eigenvectors, then applies random
    perturbations to break spatial symmetry. This is essential for
    finding the correct ground state in open-shell systems where
    symmetric solutions may be saddle points.

    Args:
        H_core: Core Hamiltonian matrix.
        X: Orthonormalization transformation.
        n_alpha: Number of alpha electrons.
        n_beta: Number of beta electrons.
        key: JAX random key (optional, defaults to PRNGKey(42)).

    Returns:
        Tuple of (C_alpha, C_beta) initial MO coefficient matrices.
    """
    # Start with core Hamiltonian eigenvectors
    C = core_hamiltonian_guess(H_core, X)

    # For closed-shell, return symmetric guess
    if n_alpha == n_beta:
        return C, C

    # For open-shell: add random perturbation to break symmetry
    if key is None:
        key = jax.random.PRNGKey(42)

    key1, key2 = jax.random.split(key)
    noise_scale = 0.1

    # Add different random noise to alpha and beta
    noise_a = jax.random.normal(key1, C.shape) * noise_scale
    noise_b = jax.random.normal(key2, C.shape) * noise_scale

    C_alpha = C + noise_a
    C_beta = C + noise_b

    # Re-orthonormalize
    C_alpha = X @ jnl.qr(jnl.solve(X, C_alpha)).Q
    C_beta = X @ jnl.qr(jnl.solve(X, C_beta)).Q

    return C_alpha, C_beta
