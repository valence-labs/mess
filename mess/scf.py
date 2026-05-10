"""Vanilla self-consistent field solver implementation."""

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnl
import equinox as eqx
from jax.lax import while_loop

from mess.basis import Basis
from mess.integrals import eri_basis, kinetic_basis, nuclear_basis, overlap_basis
from mess.structure import nuclear_energy
from mess.orthnorm import cholesky
from mess.types import OrthNormTransform

from typing import Tuple, Literal

# TODO: remove or add centralized typing.py file
FloatBxB = jnp.ndarray
FloatSCF = jnp.ndarray
FloatSCFxBxB = jnp.ndarray
FloatSCFxSCF = jnp.ndarray


def compute_residual(F: FloatBxB, P: FloatBxB, overlap: FloatBxB) -> FloatBxB:
    """
    Computes the residual matrix for the Fock matrix.
    F: Fock matrix
    P: Density matrix
    cst: constant system tensors containing the overlap matrix
    """
    temp = jnp.einsum("ab,bc,cd->ad", F, P, overlap)
    X = cholesky(overlap)  # TODO: avoid unnecessary repomputation
    res = X.T @ (temp - temp.T) @ X
    res = (res - res.T) / 2  # Recover anti-symmetry violated by numerical errors
    return res


@jax.jit
def solve_pulay_equation(current_cycle: int, overlap: FloatSCFxSCF) -> FloatSCF:
    B = overlap
    total_cycles = overlap.shape[0]
    constraint_idx = current_cycle + 1
    set_vec = -1 * (jnp.arange(total_cycles) < constraint_idx)
    B = B.at[:, constraint_idx].set(set_vec)
    B = B.at[constraint_idx, :].set(set_vec)
    B = B.at[constraint_idx, constraint_idx].set(0)
    B = (B + B.T) / 2  # Ensure symmetry
    rhs = jnp.zeros(total_cycles).at[constraint_idx].set(-1)
    fock_coeffs = jax.scipy.linalg.solve(
        B, rhs, assume_a="sym"
    )  # (x0, ..., x_{n-1}, lambda, 0, ..., 0)
    return fock_coeffs


@eqx.dataclass  # TODO: not sure wether this works, this used to be a flax dataclass
class DiisState:
    overlap: FloatBxB  # do not confuse with the basis set overlap matrix
    fock_trajectory: FloatSCFxBxB
    res_trajectory: FloatSCFxBxB

    @classmethod
    def init(
        cls,
        total_cycles: int,
        fock_matrix: FloatBxB,
        density_matrix: FloatBxB,
        overlap: FloatBxB,
    ):
        N_bas = fock_matrix.shape[0]
        # diagonal padding such that matrix is invertible and well conditioned
        overlap = jnp.diag((1 + jnp.arange(total_cycles + 1)) / (total_cycles + 1))
        fock_trajectory = jnp.zeros((total_cycles, N_bas, N_bas)).at[0].set(fock_matrix)
        residual = compute_residual(fock_matrix, density_matrix, overlap)
        res_trajectory = jnp.zeros((total_cycles, N_bas, N_bas)).at[0].set(residual)
        return cls(overlap, fock_trajectory, res_trajectory)


@jax.jit
def diis_update(
    current_cycle: int,
    raw_fock_matrix: FloatBxB,
    state: DiisState,
    density_matrix: FloatBxB,
    overlap: FloatBxB,
) -> Tuple[FloatBxB, DiisState]:
    """
    Direct Inversion of the Iterative Subspace (DIIS) to accelerate the
    convergence of the Self-Consistent Field (SCF) method.
    Returns the DIIS update to the Fock matrix.

    current_cycle: current cycle of the SCF method
    raw_fock_matrix: standard Fock matrix
    density_matrix: density matrix
    state: DIIS state based on the previous cycles
    overlap: the overlap matrix

    returns:
        (Fock matrix updated by DIIS, updated DIIS state)

    Implementation inspired by
    https://github.com/psi4/psi4numpy/blob/master/Tutorials/03_Hartree-Fock/3b_rhf-diis.ipynb
    but adapted to be jax compile friendly.
    """
    residual = compute_residual(raw_fock_matrix, density_matrix, overlap)
    i = current_cycle
    res_trajectory = state.res_trajectory.at[i].set(residual)
    new_overlap = jnp.einsum("ikl,kl->i", res_trajectory, residual)
    overlap = state.overlap.at[i, :-1].set(new_overlap)
    overlap = overlap.at[:-1, i].set(new_overlap)
    fock_coeffs = solve_pulay_equation(i, overlap)
    fock_trajectory = state.fock_trajectory.at[i].set(raw_fock_matrix)
    F_out = jnp.einsum("i,ijk->jk", fock_coeffs[:-1], fock_trajectory)
    F_out = jnp.where(
        jnp.isnan(F_out).any(), raw_fock_matrix, F_out
    )  # this is necessary, since B becomes singular once it converges converged
    return F_out, DiisState(overlap, fock_trajectory, res_trajectory)


def scf(
    basis: Basis,
    otransform: OrthNormTransform = cholesky,
    max_iters: int = 32,
    tolerance: float = 1e-4,
    method: Literal["vanilla", "diis"] = "diis",  # NOTE: momentum/dampening option?
):
    """
    Self-consistent field (SCF) method.
    TODO: integrate diis here
    """
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
