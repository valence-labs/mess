"""Many electron Hamiltonian with Density Functional Theory or Hartree-Fock."""

from typing import Callable, Literal, Optional, Tuple, get_args
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnl
import optimistix as optx
from jaxtyping import Array, ScalarLike

from mess.basis import Basis, renorm
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
    gga_exchange_b88_spinpol,
    gga_exchange_pbe,
    gga_exchange_pbe_spinpol,
    lda_correlation_vwn,
    lda_exchange,
    lda_exchange_spinpol,
)
from mess.initial_guess import symmetry_broken_guess

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
            self.overlap = jnp.array(mol.intor(f"int1e_ovlp_{kind}"))
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
        backend: IntegralBackend = "pyscf_sph",
    ):
        super().__init__()
        self.basis = renorm(basis, backend) if backend != "mess" else basis
        one_elec = OneElectron(basis, backend=backend)
        S = one_elec.overlap
        self.X = ont(S)
        self.H_core = one_elec.kinetic + one_elec.nuclear
        self.two_electron = TwoElectron(basis, backend=backend)
        self.xcfunc = build_xcfunc(xc_method, self.basis, self.two_electron)

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
        return H(P)

    solver = optx.BestSoFarMinimiser(solver)
    Z = initial_guess_fn(H.basis)
    sol = optx.minimise(f, solver, Z, max_steps=max_steps)
    C = H.orthonormalise(sol.value)
    P = H.basis.density_matrix(C)
    E_elec = H(P)
    E_total = E_elec + nuclear_energy(H.basis.structure)
    return E_total, C, sol


# =============================================================================
# Unrestricted (spin-polarized) implementations
# =============================================================================


class UnrestrictedHartreeFockExchange(eqx.Module):
    """Hartree-Fock exchange for unrestricted calculations."""

    two_electron: TwoElectron

    def __init__(self, two_electron: TwoElectron):
        self.two_electron = two_electron

    def __call__(self, P_alpha: FloatNxN, P_beta: FloatNxN) -> ScalarLike:
        """Compute HF exchange energy for alpha and beta densities."""
        K_alpha = self.two_electron.exchange(P_alpha)
        K_beta = self.two_electron.exchange(P_beta)
        # Factor of 0.5 (not 0.25) because each spin density is already separate
        return -0.5 * (jnp.sum(P_alpha * K_alpha) + jnp.sum(P_beta * K_beta))


class ULDA(eqx.Module):
    """Unrestricted LDA functional."""

    basis: Basis
    mesh: Mesh

    def __init__(self, basis: Basis):
        self.basis = basis
        self.mesh = xcmesh_from_pyscf(basis.structure)

    def __call__(self, P_alpha: FloatNxN, P_beta: FloatNxN) -> ScalarLike:
        rho_a = density(self.basis, self.mesh, P_alpha)
        rho_b = density(self.basis, self.mesh, P_beta)
        rho_total = rho_a + rho_b

        # Spin polarization with safe division (avoids NaN gradients)
        zeta = (rho_a - rho_b) / (rho_total + 1e-15)
        zeta = jnp.clip(zeta, -1.0, 1.0)

        # Exchange (spin-polarized)
        eps_x = lda_exchange_spinpol(rho_a, rho_b)

        # Correlation (with spin polarization)
        eps_c = lda_correlation_vwn(rho_total, zeta=zeta)

        E_xc = jnp.einsum("i,i,i", self.mesh.weights, rho_total, eps_x + eps_c)
        return E_xc


class UPBE(eqx.Module):
    """Unrestricted PBE functional."""

    basis: Basis
    mesh: Mesh

    def __init__(self, basis: Basis):
        self.basis = basis
        self.mesh = xcmesh_from_pyscf(basis.structure)

    def __call__(self, P_alpha: FloatNxN, P_beta: FloatNxN) -> ScalarLike:
        rho_a, grad_rho_a = density_and_grad(self.basis, self.mesh, P_alpha)
        rho_b, grad_rho_b = density_and_grad(self.basis, self.mesh, P_beta)
        rho_total = rho_a + rho_b
        grad_rho_total = grad_rho_a + grad_rho_b

        # Spin polarization with safe division (avoids NaN gradients)
        zeta = (rho_a - rho_b) / (rho_total + 1e-15)
        zeta = jnp.clip(zeta, -1.0, 1.0)

        # Exchange (spin-polarized)
        eps_x = gga_exchange_pbe_spinpol(rho_a, rho_b, grad_rho_a, grad_rho_b)

        # Correlation (with spin polarization)
        eps_c = gga_correlation_pbe(rho_total, grad_rho_total, zeta=zeta)

        E_xc = jnp.einsum("i,i,i", self.mesh.weights, rho_total, eps_x + eps_c)
        return E_xc


class UPBE0(eqx.Module):
    """Unrestricted PBE0 hybrid functional."""

    basis: Basis
    mesh: Mesh
    hfx: UnrestrictedHartreeFockExchange

    def __init__(self, basis: Basis, two_electron: TwoElectron):
        self.basis = basis
        self.mesh = xcmesh_from_pyscf(basis.structure)
        self.hfx = UnrestrictedHartreeFockExchange(two_electron)

    def __call__(self, P_alpha: FloatNxN, P_beta: FloatNxN) -> ScalarLike:
        rho_a, grad_rho_a = density_and_grad(self.basis, self.mesh, P_alpha)
        rho_b, grad_rho_b = density_and_grad(self.basis, self.mesh, P_beta)
        rho_total = rho_a + rho_b
        grad_rho_total = grad_rho_a + grad_rho_b

        # Spin polarization with safe division (avoids NaN gradients)
        zeta = (rho_a - rho_b) / (rho_total + 1e-15)
        zeta = jnp.clip(zeta, -1.0, 1.0)

        eps_x = 0.75 * gga_exchange_pbe_spinpol(rho_a, rho_b, grad_rho_a, grad_rho_b)
        eps_c = gga_correlation_pbe(rho_total, grad_rho_total, zeta=zeta)

        E_xc = jnp.einsum("i,i,i", self.mesh.weights, rho_total, eps_x + eps_c)
        return E_xc + 0.25 * self.hfx(P_alpha, P_beta)


class UB3LYP(eqx.Module):
    """Unrestricted B3LYP hybrid functional."""

    basis: Basis
    mesh: Mesh
    hfx: UnrestrictedHartreeFockExchange

    def __init__(self, basis: Basis, two_electron: TwoElectron):
        self.basis = basis
        self.mesh = xcmesh_from_pyscf(basis.structure)
        self.hfx = UnrestrictedHartreeFockExchange(two_electron)

    def __call__(self, P_alpha: FloatNxN, P_beta: FloatNxN) -> ScalarLike:
        rho_a, grad_rho_a = density_and_grad(self.basis, self.mesh, P_alpha)
        rho_b, grad_rho_b = density_and_grad(self.basis, self.mesh, P_beta)
        rho_total = rho_a + rho_b

        # Spin polarization with safe division (avoids NaN gradients)
        zeta = (rho_a - rho_b) / (rho_total + 1e-15)
        zeta = jnp.clip(zeta, -1.0, 1.0)

        # B3LYP exchange: 0.08*LDA + 0.72*B88
        eps_x_lda = lda_exchange_spinpol(rho_a, rho_b)
        eps_x_b88 = gga_exchange_b88_spinpol(rho_a, rho_b, grad_rho_a, grad_rho_b)
        eps_x = 0.08 * eps_x_lda + 0.72 * eps_x_b88

        # B3LYP correlation: 0.19*VWN + 0.81*LYP
        vwn_c = (1 - 0.81) * lda_correlation_vwn(rho_total, zeta=zeta)
        # Note: LYP correlation doesn't have simple spin-polarized form in original
        # Using restricted LYP as approximation (common practice)
        lyp_c = 0.81 * gga_correlation_lyp(rho_total, grad_rho_a + grad_rho_b)

        E_xc = jnp.einsum("i,i,i", self.mesh.weights, rho_total, eps_x + vwn_c + lyp_c)
        return E_xc + 0.2 * self.hfx(P_alpha, P_beta)


def build_xcfunc_unrestricted(
    xc_method: xcstr, basis: Basis, two_electron: Optional[TwoElectron] = None
) -> eqx.Module:
    """Build an unrestricted XC functional."""
    if two_electron is None and xc_method in ("pbe0", "b3lyp"):
        raise ValueError(
            f"Hybrid functional {xc_method} requires providing TwoElectron integrals"
        )

    match xc_method:
        case "lda":
            return ULDA(basis)
        case "pbe":
            return UPBE(basis)
        case "pbe0":
            return UPBE0(basis, two_electron)
        case "b3lyp":
            return UB3LYP(basis, two_electron)
        case "hfx":
            return UnrestrictedHartreeFockExchange(two_electron)
        case _:
            methods = get_args(xcstr)
            methods = ", ".join(methods)
            msg = f"Unsupported exchange-correlation option: {xc_method}."
            msg += f"\nMust be one of the following: {methods}"
            raise ValueError(msg)


class UHamiltonian(eqx.Module):
    """Unrestricted Hamiltonian for spin-polarized calculations."""

    X: FloatNxN
    S: FloatNxN
    H_core: FloatNxN
    basis: Basis
    two_electron: TwoElectron
    xcfunc: eqx.Module

    def __init__(
        self,
        basis: Basis,
        ont: OrthNormTransform = symmetric,
        xc_method: xcstr = "lda",
        backend: IntegralBackend = "pyscf_sph",
    ):
        super().__init__()
        self.basis = renorm(basis, backend) if backend != "mess" else basis
        one_elec = OneElectron(basis, backend=backend)
        self.S = one_elec.overlap
        self.X = ont(self.S)
        self.H_core = one_elec.kinetic + one_elec.nuclear
        self.two_electron = TwoElectron(basis, backend=backend)
        self.xcfunc = build_xcfunc_unrestricted(xc_method, self.basis, self.two_electron)

    def __call__(self, P_alpha: FloatNxN, P_beta: FloatNxN) -> ScalarLike:
        """Compute electronic energy from alpha and beta density matrices."""
        P_total = P_alpha + P_beta

        # One-electron energy
        E_core = jnp.sum(self.H_core * P_total)

        # Coulomb energy from total density
        J = self.two_electron.coloumb(P_total)
        E_J = 0.5 * jnp.sum(J * P_total)

        # Exchange-correlation energy
        E_xc = self.xcfunc(P_alpha, P_beta)

        return E_core + E_J + E_xc

    def orthonormalise(self, Z: FloatNxN) -> FloatNxN:
        """Orthonormalize coefficient matrix."""
        C = self.X @ jnl.qr(Z).Q
        return C


@partial(jax.jit, static_argnames=("max_steps",))
def _uminimise_inner(
    H: UHamiltonian,
    Z_init: FloatNxN,
    max_steps: Optional[int] = None,
) -> Tuple[ScalarLike, FloatNxN, FloatNxN, optx.Solution]:
    """Inner JIT-compiled optimization loop for unrestricted SCF."""
    n = H.basis.num_orbitals

    def f(Z_concat, _):
        # Split concatenated matrix into alpha and beta
        Z_alpha = Z_concat[:, :n]
        Z_beta = Z_concat[:, n:]

        C_alpha = H.orthonormalise(Z_alpha)
        C_beta = H.orthonormalise(Z_beta)

        P_alpha = H.basis.density_matrix_alpha(C_alpha)
        P_beta = H.basis.density_matrix_beta(C_beta)

        return H(P_alpha, P_beta)

    solver = optx.BestSoFarMinimiser(optx.BFGS(atol=1e-6, rtol=1e-5))

    sol = optx.minimise(f, solver, Z_init, max_steps=max_steps)

    # Extract final coefficients
    Z_alpha = sol.value[:, :n]
    Z_beta = sol.value[:, n:]
    C_alpha = H.orthonormalise(Z_alpha)
    C_beta = H.orthonormalise(Z_beta)

    P_alpha = H.basis.density_matrix_alpha(C_alpha)
    P_beta = H.basis.density_matrix_beta(C_beta)

    E_elec = H(P_alpha, P_beta)
    E_total = E_elec + nuclear_energy(H.basis.structure)

    return E_total, C_alpha, C_beta, sol


def uminimise(
    H: UHamiltonian,
    max_steps: Optional[int] = None,
) -> Tuple[ScalarLike, FloatNxN, FloatNxN, optx.Solution]:
    """Solve for the electronic coefficients that minimise the unrestricted energy.

    Args:
        H: The unrestricted Hamiltonian.
        max_steps: Maximum number of minimizer steps.

    Returns:
        Tuple containing:
            - total energy in atomic units
            - alpha coefficient matrix C_alpha
            - beta coefficient matrix C_beta
            - the optimistix.Solution object
    """
    # Use symmetry-broken guess for better convergence on open-shell systems
    # Computed outside JIT since it may use random numbers
    C_alpha, C_beta = symmetry_broken_guess(
        H.H_core, H.X,
        H.basis.structure.n_alpha,
        H.basis.structure.n_beta,
    )
    Z_init = jnp.concatenate([C_alpha, C_beta], axis=1)

    return _uminimise_inner(H, Z_init, max_steps)
