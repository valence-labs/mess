"""Many electron Hamiltonian with Density Functional Theory or Hartree-Fock."""

from typing import Literal, Optional, Tuple, get_args
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnl
import optimistix as optx
from jaxtyping import Array, ScalarLike

from mess.basis import Basis
from mess.integrals import eri_basis, kinetic_basis, nuclear_basis, overlap_basis
from mess.interop import to_pyscf
from mess.mesh import Mesh, density, density_and_grad, xcmesh_from_pyscf
from mess.orthnorm import symmetric
from mess.structure import nuclear_energy
from mess.types import FloatNxN, OrthNormTransform
from mess.xcfunctional import (
    gga_correlation_lyp,
    gga_correlation_pbe,
    gga_exchange_b88,
    gga_exchange_pbe,
    lda_correlation_vwn,
    lda_exchange,
)

xcstr = Literal["lda", "pbe", "pbe0", "b3lyp", "hfx"]
IntegralBackend = Literal["mess", "pyscf_cart", "pyscf_sph"]


class OneElectron(eqx.Module):
    overlap: FloatNxN
    kinetic: FloatNxN
    nuclear: FloatNxN

    def __init__(self, basis: Basis, backend: IntegralBackend = "mess"):
        """_summary_

        Args:
            basis (Basis): _description_
            backend (IntegralBackend, optional): _description_. Defaults to "mess".

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if backend == "mess":
            self.overlap = overlap_basis(basis)
            self.kinetic = kinetic_basis(basis)
            self.nuclear = nuclear_basis(basis).sum(axis=0)
        elif backend.startswith("pyscf_"):
            mol = to_pyscf(basis.structure, basis.basis_name)
            kind = backend.split("_")[1]
            S = jnp.array(mol.intor(f"int1e_ovlp_{kind}"))
            N = 1 / jnp.sqrt(jnp.diagonal(S))
            self.overlap = N[:, jnp.newaxis] * N[jnp.newaxis, :] * S
            self.kinetic = jnp.array(mol.intor(f"int1e_kin_{kind}"))
            self.nuclear = jnp.array(mol.intor(f"int1e_nuc_{kind}"))


class TwoElectron(eqx.Module):
    eri: Array

    def __init__(self, basis: Basis, backend: str = "mess"):
        """

        Args:
            basis (Basis): the basis set used to build the electron repulsion integrals
            backend (str, optional): Integral backend used. Defaults to "mess".
        """
        super().__init__()
        if backend == "mess":
            self.eri = eri_basis(basis)
        elif backend.startswith("pyscf_"):
            mol = to_pyscf(basis.structure, basis.basis_name)
            kind = backend.split("_")[1]
            self.eri = jnp.array(mol.intor(f"int2e_{kind}", aosym="s1"))

    def coloumb(self, P: FloatNxN) -> FloatNxN:
        """Build the Coloumb matrix (classical electrostatic) from the density matrix.

        Args:
            P (FloatNxN): the density matrix

        Returns:
            FloatNxN: Coloumb matrix
        """
        return jnp.einsum("kl,ijkl->ij", P, self.eri)

    def exchange(self, P: FloatNxN) -> FloatNxN:
        """Build the quantum-mechanical exchange matrix from the density matrix

        Args:
            P (FloatNxN): the density matrix

        Returns:
            FloatNxN: Exchange matrix
        """
        return jnp.einsum("ij,ikjl->kl", P, self.eri)


class HartreeFockExchange(eqx.Module):
    two_electron: TwoElectron

    def __init__(self, two_electron: TwoElectron):
        self.two_electron = two_electron

    def __call__(self, P: FloatNxN) -> ScalarLike:
        K = self.two_electron.exchange(P)
        return -0.25 * jnp.sum(P * K)


class LDA(eqx.Module):
    basis: Basis
    mesh: Mesh

    def __init__(self, basis: Basis):
        self.basis = basis
        self.mesh = xcmesh_from_pyscf(basis.structure)

    def __call__(self, P: FloatNxN) -> ScalarLike:
        rho = density(self.basis, self.mesh, P)
        eps_xc = lda_exchange(rho) + lda_correlation_vwn(rho)
        E_xc = jnp.einsum("i,i,i", self.mesh.weights, rho, eps_xc)
        return E_xc


class PBE(eqx.Module):
    basis: Basis
    mesh: Mesh

    def __init__(self, basis: Basis):
        self.basis = basis
        self.mesh = xcmesh_from_pyscf(basis.structure)

    def __call__(self, P: FloatNxN) -> ScalarLike:
        rho, grad_rho = density_and_grad(self.basis, self.mesh, P)
        eps_xc = gga_exchange_pbe(rho, grad_rho) + gga_correlation_pbe(rho, grad_rho)
        E_xc = jnp.einsum("i,i,i", self.mesh.weights, rho, eps_xc)
        return E_xc


class PBE0(eqx.Module):
    basis: Basis
    mesh: Mesh
    hfx: HartreeFockExchange

    def __init__(self, basis: Basis, two_electron: TwoElectron):
        self.basis = basis
        self.mesh = xcmesh_from_pyscf(basis.structure)
        self.hfx = HartreeFockExchange(two_electron)

    def __call__(self, P: FloatNxN) -> ScalarLike:
        rho, grad_rho = density_and_grad(self.basis, self.mesh, P)
        e = 0.75 * gga_exchange_pbe(rho, grad_rho) + gga_correlation_pbe(rho, grad_rho)
        E_xc = jnp.einsum("i,i,i", self.mesh.weights, rho, e)
        return E_xc + 0.25 * self.hfx(P)


class B3LYP(eqx.Module):
    basis: Basis
    mesh: Mesh
    hfx: HartreeFockExchange

    def __init__(self, basis: Basis, two_electron: TwoElectron):
        self.basis = basis
        self.mesh = xcmesh_from_pyscf(basis.structure)
        self.hfx = HartreeFockExchange(two_electron)

    def __call__(self, P: FloatNxN) -> ScalarLike:
        rho, grad_rho = density_and_grad(self.basis, self.mesh, P)
        eps_x = 0.08 * lda_exchange(rho) + 0.72 * gga_exchange_b88(rho, grad_rho)
        vwn_c = (1 - 0.81) * lda_correlation_vwn(rho)
        lyp_c = 0.81 * gga_correlation_lyp(rho, grad_rho)
        b3lyp_xc = eps_x + vwn_c + lyp_c
        E_xc = jnp.einsum("i,i,i", self.mesh.weights, rho, b3lyp_xc)
        return E_xc + 0.2 * self.hfx(P)


def build_xcfunc(
    xc_method: xcstr, basis: Basis, two_electron: Optional[TwoElectron] = None
) -> eqx.Module:
    if two_electron is None and xc_method in ("pbe0", "b3lyp"):
        raise ValueError(
            f"Hybrid functional {xc_method} requires providing TwoElectron integrals"
        )

    match xc_method:
        case "lda":
            return LDA(basis)
        case "pbe":
            return PBE(basis)
        case "pbe0":
            return PBE0(basis, two_electron)
        case "b3lyp":
            return B3LYP(basis, two_electron)
        case "hfx":
            return HartreeFockExchange(two_electron)
        case _:
            methods = get_args(xcstr)
            methods = ", ".join(methods)
            msg = f"Unsupported exchange-correlation option: {xc_method}."
            msg += f"\nMust be one of the following: {methods}"
            raise ValueError(msg)


class Hamiltonian(eqx.Module):
    X: FloatNxN
    H_core: FloatNxN
    basis: Basis
    two_electron: TwoElectron
    xcfunc: eqx.Module

    def __init__(
        self,
        basis: Basis,
        ont: OrthNormTransform = symmetric,
        xc_method: xcstr = "lda",
        backend: IntegralBackend = "pyscf_cart",
    ):
        super().__init__()
        self.basis = basis
        one_elec = OneElectron(basis, backend=backend)
        S = one_elec.overlap
        self.X = ont(S)
        self.H_core = one_elec.kinetic + one_elec.nuclear
        self.two_electron = TwoElectron(basis, backend=backend)
        self.xcfunc = build_xcfunc(xc_method, basis, self.two_electron)

    def __call__(self, P: FloatNxN) -> ScalarLike:
        E_core = jnp.sum(self.H_core * P)
        E_xc = self.xcfunc(P)
        J = self.two_electron.coloumb(P)
        E_es = 0.5 * jnp.sum(J * P)
        E = E_core + E_xc + E_es
        return E

    def orthonormalise(self, Z: FloatNxN) -> FloatNxN:
        C = self.X @ jnl.qr(Z).Q
        return C


@partial(jax.jit, static_argnames=("max_steps", "atol", "rtol"))
def minimise(
    H: Hamiltonian,
    max_steps: Optional[int] = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> Tuple[ScalarLike, FloatNxN, optx.Solution]:
    """Solve for the electronic coefficients that minimise the total energy

    This function takes a Hamiltonian built for a given basis set and molecular
    structure, and finds the electronic coefficients that minimise the total energy.
    The optimisation is performed using the BFGS algorithm.

    Args:
        H (Hamiltonian): The Hamiltonian for a given basis set and molecular structure.
        max_steps (Optional[int]): Maximum number of minimizer steps. Defaults to None.
        atol (float): Absolute tolerance for convergence. Defaults to 1e-6.
        rtol (float): Relative tolerance for convergence. Defaults to 1e-5.

    Returns:
        Tuple[ScalarLike, FloatNxN, optimistix.Solution]: A tuple containing:
            - total energy in atomic units
            - coefficient matrix C that minimizes the Hamiltonian
            - the optimistix.Solution object
    """

    def f(Z, _):
        C = H.orthonormalise(Z)
        P = H.basis.density_matrix(C)
        return H(P)

    solver = optx.BFGS(atol=atol, rtol=rtol)
    Z = jnp.eye(H.basis.num_orbitals)
    sol = optx.minimise(f, solver, Z, max_steps=max_steps)
    C = H.orthonormalise(sol.value)
    P = H.basis.density_matrix(C)
    E_elec = H(P)
    E_total = E_elec + nuclear_energy(H.basis.structure)
    return E_total, C, sol
