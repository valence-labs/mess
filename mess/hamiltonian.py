"""Many electron Hamiltonian with Density Functional Theory or Hartree-Fock."""

from typing import Callable, Literal, Optional, Tuple, get_args
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnl
import optimistix as optx
from jaxtyping import ScalarLike

from mess.basis import Basis, renorm
from mess.integrals import kinetic_basis, nuclear_basis, overlap_basis
from mess.interop import to_pyscf
from mess.mesh import Mesh, density, density_and_grad, xcmesh_from_pyscf
from mess.orthnorm import symmetric
from mess.structure import nuclear_energy
from mess.two_electron import TwoElectron, ri_from_basis, isdf_thc_ri
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
CoulombMethod = Literal["full", "ri", "thc-ri"]


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
            self.overlap = jnp.array(mol.intor(f"int1e_ovlp_{kind}"))
            self.kinetic = jnp.array(mol.intor(f"int1e_kin_{kind}"))
            self.nuclear = jnp.array(mol.intor(f"int1e_nuc_{kind}"))


class HartreeFockExchange(eqx.Module):
    two_electron: eqx.Module

    def __init__(self, two_electron: eqx.Module):
        self.two_electron = two_electron

    def __call__(self, P: FloatNxN, C_occ=None) -> ScalarLike:
        K = self.two_electron.exchange(P, C_occ)
        return -0.25 * jnp.sum(P * K)


class LDA(eqx.Module):
    basis: Basis
    mesh: Mesh

    def __init__(self, basis: Basis):
        self.basis = basis
        self.mesh = xcmesh_from_pyscf(basis.structure)

    def __call__(self, P: FloatNxN, C_occ=None) -> ScalarLike:
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

    def __call__(self, P: FloatNxN, C_occ=None) -> ScalarLike:
        rho, grad_rho = density_and_grad(self.basis, self.mesh, P)
        eps_xc = gga_exchange_pbe(rho, grad_rho) + gga_correlation_pbe(rho, grad_rho)
        E_xc = jnp.einsum("i,i,i", self.mesh.weights, rho, eps_xc)
        return E_xc


class PBE0(eqx.Module):
    basis: Basis
    mesh: Mesh
    hfx: HartreeFockExchange

    def __init__(self, basis: Basis, two_electron: eqx.Module):
        self.basis = basis
        self.mesh = xcmesh_from_pyscf(basis.structure)
        self.hfx = HartreeFockExchange(two_electron)

    def __call__(self, P: FloatNxN, C_occ=None) -> ScalarLike:
        rho, grad_rho = density_and_grad(self.basis, self.mesh, P)
        e = 0.75 * gga_exchange_pbe(rho, grad_rho) + gga_correlation_pbe(rho, grad_rho)
        E_xc = jnp.einsum("i,i,i", self.mesh.weights, rho, e)
        return E_xc + 0.25 * self.hfx(P, C_occ)


class B3LYP(eqx.Module):
    basis: Basis
    mesh: Mesh
    hfx: HartreeFockExchange

    def __init__(self, basis: Basis, two_electron: eqx.Module):
        self.basis = basis
        self.mesh = xcmesh_from_pyscf(basis.structure)
        self.hfx = HartreeFockExchange(two_electron)

    def __call__(self, P: FloatNxN, C_occ=None) -> ScalarLike:
        rho, grad_rho = density_and_grad(self.basis, self.mesh, P)
        eps_x = 0.08 * lda_exchange(rho) + 0.72 * gga_exchange_b88(rho, grad_rho)
        vwn_c = (1 - 0.81) * lda_correlation_vwn(rho)
        lyp_c = 0.81 * gga_correlation_lyp(rho, grad_rho)
        b3lyp_xc = eps_x + vwn_c + lyp_c
        E_xc = jnp.einsum("i,i,i", self.mesh.weights, rho, b3lyp_xc)
        return E_xc + 0.2 * self.hfx(P, C_occ)


def build_xcfunc(
    xc_method: xcstr, basis: Basis, two_electron: Optional[eqx.Module] = None
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
    two_electron: eqx.Module
    xcfunc: eqx.Module

    def __init__(
        self,
        basis: Basis,
        ont: OrthNormTransform = symmetric,
        xc_method: xcstr = "lda",
        backend: IntegralBackend = "pyscf_sph",
        coulomb: CoulombMethod = "full",
        two_electron: Optional[eqx.Module] = None,
    ):
        super().__init__()
        self.basis = renorm(basis, backend) if backend != "mess" else basis
        one_elec = OneElectron(basis, backend=backend)
        S = one_elec.overlap
        self.X = ont(S)
        self.H_core = one_elec.kinetic + one_elec.nuclear
        if two_electron is not None:
            self.two_electron = two_electron
        else:
            match coulomb:
                case "full":
                    self.two_electron = TwoElectron(basis, backend=backend)
                case "ri":
                    self.two_electron = ri_from_basis(basis)
                case "thc-ri":
                    mesh = xcmesh_from_pyscf(basis.structure, level=0)
                    self.two_electron = isdf_thc_ri(basis, mesh, c_isdf=3.0)
                case _:
                    methods = get_args(CoulombMethod)
                    raise ValueError(
                        f"Unknown coulomb method: {coulomb}. "
                        f"Must be one of: {', '.join(methods)}"
                    )
        self.xcfunc = build_xcfunc(xc_method, self.basis, self.two_electron)

    def __call__(self, P: FloatNxN, C_occ=None) -> ScalarLike:
        E_core = jnp.sum(self.H_core * P)
        E_xc = self.xcfunc(P, C_occ)
        J = self.two_electron.coloumb(P)
        E_es = 0.5 * jnp.sum(J * P)
        E = E_core + E_xc + E_es
        return E

    def orthonormalise(self, Z: FloatNxN) -> FloatNxN:
        C = self.X @ jnl.qr(Z).Q
        return C


def identity_guess(basis: Basis) -> FloatNxN:
    return jnp.eye(basis.num_orbitals)


@partial(jax.jit, static_argnames=("max_steps"))
def minimise(
    H: Hamiltonian,
    max_steps: Optional[int] = None,
    solver: optx.AbstractMinimiser = optx.BFGS(atol=1e-6, rtol=1e-5),
    initial_guess_fn: Callable[[Basis], FloatNxN] = identity_guess,
) -> Tuple[ScalarLike, FloatNxN, optx.Solution]:
    """Solve for the electronic coefficients that minimise the total energy

    This function takes a Hamiltonian built for a given basis set and molecular
    structure, and finds the electronic coefficients that minimise the total energy.
    The optimisation is performed using the BFGS algorithm.

    Args:
        H (Hamiltonian): The Hamiltonian for a given basis set and molecular structure.
        max_steps (Optional[int]): Maximum number of minimizer steps. Defaults to None.
        solver (optimistix.AbstractMinimizer): Solver instance to use to minimise the
            electronic energy. Defaults to BFGS.
        initial_guess_fn (Callable[[Basis], FloatNxN]): A function that provides an
            initial guess for the optimization matrix. This is then orthonormalized to
            form the molecular orbital coefficients. Defaults to `identity_guess`.

    Returns:
        Tuple[ScalarLike, FloatNxN, optimistix.Solution]: A tuple containing:
            - total energy in atomic units
            - coefficient matrix C that minimizes the Hamiltonian
            - the optimistix.Solution object
    """

    def f(Z, _):
        C = H.orthonormalise(Z)
        P = H.basis.density_matrix(C)
        n_occ = H.basis.structure.num_electrons // 2
        C_occ = C[:, :n_occ]
        return H(P, C_occ=C_occ)

    solver = optx.BestSoFarMinimiser(solver)
    Z = initial_guess_fn(H.basis)
    sol = optx.minimise(f, solver, Z, max_steps=max_steps)
    C = H.orthonormalise(sol.value)
    P = H.basis.density_matrix(C)
    n_occ = H.basis.structure.num_electrons // 2
    C_occ = C[:, :n_occ]
    E_elec = H(P, C_occ=C_occ)
    E_total = E_elec + nuclear_energy(H.basis.structure)
    return E_total, C, sol
