"""Vanilla self-consistent field solver implementation."""

import jax.numpy as jnp
import jax.numpy.linalg as jnl
from jax.lax import while_loop

from mess.basis import Basis
from mess.integrals import eri_basis, kinetic_basis, nuclear_basis, overlap_basis
from mess.structure import nuclear_energy
from mess.orthnorm import cholesky
from mess.types import OrthNormTransform


def scf(
    basis: Basis,
    otransform: OrthNormTransform = cholesky,
    max_iters: int = 32,
    tolerance: float = 1e-4,
):
    """ """
    # init
    Hcore = kinetic_basis(basis) + nuclear_basis(basis).sum(axis=0)
    S = overlap_basis(basis)
    eri = eri_basis(basis)

    # initial guess for MO coeffs
    X = otransform(S)
    C = X @ jnl.eigh(X.T @ Hcore @ X)[1]

    # setup self-consistent iteration as a while loop
    counter = 0
    E = 0.0
    E_prev = 2 * tolerance
    scf_args = (counter, E, E_prev, C)

    def while_cond(scf_args):
        counter, E, E_prev, _ = scf_args
        return (counter < max_iters) & (jnp.abs(E - E_prev) > tolerance)

    def while_body(scf_args):
        counter, E, E_prev, C = scf_args
        P = basis.occupancy * C @ C.T
        J = jnp.einsum("kl,ijkl->ij", P, eri)
        K = jnp.einsum("ij,ikjl->kl", P, eri)
        G = J - 0.5 * K
        H = Hcore + G
        C = X @ jnl.eigh(X.T @ H @ X)[1]
        E_prev = E
        E = 0.5 * jnp.sum(Hcore * P) + 0.5 * jnp.sum(H * P)
        return (counter + 1, E, E_prev, C)

    _, E_electronic, _, _ = while_loop(while_cond, while_body, scf_args)
    E_nuclear = nuclear_energy(basis.structure)
    return E_nuclear + E_electronic
