"""Two-electron integral representations: full, RI, and THC-ISDF.

Provides three factorizations with both Coulomb (J) and exchange (K) builds:

- TwoElectron: full O(N^4) ERI tensor
- TwoElectronRI: Resolution-of-Identity, O(N^2 * N_aux)
- TwoElectronTHC: Tensor Hypercontraction via ISDF, O(N^2 * M + N * M^2)

References:
    Lee, Lin, Head-Gordon, arXiv:1911.00470
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from scipy.linalg import cholesky, qr, solve_triangular

from mess.basis import Basis
from mess.integrals import eri_basis
from mess.interop import to_pyscf
from mess.mesh import Mesh, xcmesh_from_pyscf
from mess.types import FloatNxM, FloatNxN


class TwoElectron(eqx.Module):
    eri: Array

    def __init__(self, basis: Basis, backend: str = "mess"):
        """
        Args:
            basis: the basis set used to build the electron repulsion integrals
            backend: Integral backend used. Defaults to "mess".
        """
        super().__init__()
        if backend == "mess":
            self.eri = eri_basis(basis)
        elif backend.startswith("pyscf_"):
            mol = to_pyscf(basis.structure, basis.basis_name)
            kind = backend.split("_")[1]
            self.eri = jnp.array(mol.intor(f"int2e_{kind}", aosym="s1"))

    def coloumb(self, P: FloatNxN) -> FloatNxN:
        """Build the Coulomb matrix from the density matrix.

        Args:
            P: the density matrix

        Returns:
            Coulomb matrix
        """
        return jnp.einsum("kl,ijkl->ij", P, self.eri)

    def exchange(self, P: FloatNxN) -> FloatNxN:
        """Build the exchange matrix from the density matrix.

        Args:
            P: the density matrix

        Returns:
            Exchange matrix
        """
        return jnp.einsum("ij,ikjl->kl", P, self.eri)


class TwoElectronRI(eqx.Module):
    """RI-J/K factored two-electron integrals.

    Stores Cholesky-contracted RI coefficients B (N_aux, N, N) where
    B^P_{mn} = sum_Q L^{-1}_{PQ} (Q|mn).

    Coulomb and exchange build cost: O(N^2 * N_aux).
    """

    B: Array  # (N_aux, N, N)

    def coloumb(self, P: FloatNxN) -> FloatNxN:
        """Build the Coulomb matrix from the density matrix using RI factors.

        Args:
            P: density matrix (N, N)

        Returns:
            Coulomb matrix J (N, N)
        """
        c = jnp.einsum("Pmn,mn->P", self.B, P)
        J = jnp.einsum("Pij,P->ij", self.B, c)
        return J

    def exchange(self, P: FloatNxN) -> FloatNxN:
        """Build the exchange matrix from the density matrix using RI factors.

        Args:
            P: density matrix (N, N)

        Returns:
            Exchange matrix K (N, N)
        """
        D = jnp.einsum("ij,Pjl->Pil", P, self.B)   # (N_aux, N, N)
        K = jnp.einsum("Pik,Pil->kl", self.B, D)    # (N, N)
        return K


class TwoElectronTHC(eqx.Module):
    """THC-ISDF factored two-electron integrals.

    Stores collocation matrix X (N, M) and core Z (M, M).
    The ERI is approximated as:

        (ij|kl) ~ sum_PQ X[i,P] X[j,P] Z[P,Q] X[k,Q] X[l,Q]

    Coulomb build: O(NM + M^2).
    Exchange build: O(N^2 M + NM^2).  (Eq. 38 of arXiv:1911.00470)
    """

    X: FloatNxM
    Z: Array

    def coloumb(self, P: FloatNxN) -> FloatNxN:
        """Build the Coulomb matrix from the density matrix using THC factors.

        Args:
            P: density matrix (N, N)

        Returns:
            Coulomb matrix J (N, N)
        """
        rho = jnp.einsum("kP,lP,kl->P", self.X, self.X, P)
        v = self.Z @ rho
        return jnp.einsum("iP,P,jP->ij", self.X, v, self.X)

    def exchange(self, P: FloatNxN) -> FloatNxN:
        """Build the exchange matrix from the density matrix using THC factors.

        AO-THC-K, Eq. 38 of Lee, Lin, Head-Gordon (arXiv:1911.00470).

        Args:
            P: density matrix (N, N)

        Returns:
            Exchange matrix K (N, N)
        """
        G = jnp.einsum("iP,ij,jQ->PQ", self.X, P, self.X)  # (M, M)
        return jnp.einsum("kP,PQ,lQ->kl", self.X, self.Z * G, self.X)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_ri_integrals(basis: Basis, auxbasis: str | None = None) -> np.ndarray:
    """Compute Cholesky-contracted RI coefficients from three-center integrals.

    Uses PySCF's df module to compute:
        1. Three-center integrals (mn|P) via int3c2e
        2. Two-center integrals (P|Q) via int2c2e
        3. Cholesky decompose (P|Q) = L L^T
        4. Solve L^{-1} (P|mn) to get B^P_{mn}

    Args:
        basis: the Basis for AO integrals
        auxbasis: auxiliary basis set name (None = PySCF auto-selects)

    Returns:
        B: ndarray (N_aux, N, N) of Cholesky-contracted RI coefficients
    """
    from pyscf import df

    mol = to_pyscf(basis.structure, basis.basis_name)
    auxmol = df.addons.make_auxmol(mol, auxbasis)

    # Three-center integrals: (N, N, N_aux)
    int3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e")
    N = mol.nao_nr()
    N_aux = auxmol.nao_nr()
    int3c = int3c.reshape(N, N, N_aux)

    # Two-center integrals: (N_aux, N_aux)
    int2c = auxmol.intor("int2c2e")

    # Cholesky decompose (P|Q) = L L^T, then solve L B = int3c for B = L^{-1} int3c
    L = cholesky(int2c, lower=True)
    int3c_flat = int3c.reshape(N * N, N_aux).T  # (N_aux, N*N)
    B_flat = solve_triangular(L, int3c_flat, lower=True)  # (N_aux, N*N)
    B = B_flat.reshape(N_aux, N, N)

    return B


def _isdf_select_points(
    basis: Basis, mesh: Mesh, n_interp: int
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate AOs on mesh and select interpolation points via QRCP.

    Steps:
        1. Evaluate AOs on mesh grid points -> phi (G, N)
        2. Form pair products zeta (G, N^2) = phi[:,i] * phi[:,j]
        3. Column-pivoted QR on zeta.T to select n_interp points

    Args:
        basis: the Basis to evaluate AOs
        mesh: quadrature mesh providing grid points
        n_interp: number of interpolation points

    Returns:
        phi: AO values on grid (G, N) as numpy array
        selected: indices of selected grid points (n_interp,)
    """
    N = basis.num_orbitals
    phi = np.asarray(basis(mesh.points))  # (G, N)
    zeta = (phi[:, :, None] * phi[:, None, :]).reshape(phi.shape[0], N * N)
    _, _, piv = qr(zeta.T, pivoting=True)
    selected = piv[:n_interp]
    return phi, selected


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------

def ri_from_basis(basis: Basis, auxbasis: str | None = None) -> TwoElectronRI:
    """Factory returning a TwoElectronRI from a Basis.

    Args:
        basis: the Basis for AO integrals
        auxbasis: auxiliary basis set name (None = PySCF auto-selects)

    Returns:
        TwoElectronRI with Cholesky-contracted RI coefficients
    """
    B = _compute_ri_integrals(basis, auxbasis)
    return TwoElectronRI(B=jnp.array(B))


def isdf_thc(
    basis: Basis, mesh: Mesh, eri: Array, n_interp: int | None = None
) -> TwoElectronTHC:
    """Fit THC-ISDF factors from a reference ERI tensor.

    Steps:
        1. QRCP to select interpolation points
        2. Build collocation matrix X = phi[selected, :].T  -> (N, M)
        3. Fit Z via double pseudoinverse against reference ERIs

    Note:
        This function requires the full O(N^4) ERI tensor. For a more efficient
        path that only needs O(N^2 * N_aux) three-center integrals, see
        :func:`isdf_thc_ri`.

    Args:
        basis: the Basis to evaluate AOs
        mesh: quadrature mesh providing grid points
        eri: reference ERI tensor (N, N, N, N)
        n_interp: number of interpolation points (default: 20 * N_basis)

    Returns:
        TwoElectronTHC with fitted X and Z
    """
    N = basis.num_orbitals
    if n_interp is None:
        n_interp = 20 * N

    phi, selected = _isdf_select_points(basis, mesh, n_interp)

    # Collocation matrix
    X = jnp.array(phi[selected, :].T)  # (N, M)

    # Fit Z: B[ij, P] = X[i,P] * X[j,P], then Z = pinv(B) @ eri_mat @ pinv(B).T
    B = (X[:, None, :] * X[None, :, :]).reshape(N * N, n_interp)  # (N^2, M)
    B_np = np.asarray(B)
    B_pinv = np.linalg.pinv(B_np)  # (M, N^2)
    eri_mat = np.asarray(eri).reshape(N * N, N * N)
    Z = jnp.array(B_pinv @ eri_mat @ B_pinv.T)  # (M, M)

    return TwoElectronTHC(X=X, Z=Z)


def isdf_thc_ri(
    basis: Basis,
    mesh: Mesh,
    auxbasis: str | None = None,
    c_isdf: float = 5.0,
) -> TwoElectronTHC:
    """Fit THC-ISDF factors from RI integrals, never forming the full ERI.

    Following Lee, Lin, Head-Gordon (JCTC 2020, 16, 243), the number of
    interpolation points is N_IP = c_ISDF * N_X, where N_X is the auxiliary
    basis size.

    Args:
        basis: the Basis to evaluate AOs
        mesh: quadrature mesh providing grid points
        auxbasis: auxiliary basis set name (None = PySCF auto-selects)
        c_isdf: ISDF interpolation ratio; N_IP = c_ISDF * N_aux (default: 5.0)

    Returns:
        TwoElectronTHC with fitted X and Z
    """
    N = basis.num_orbitals

    # 1. Compute RI coefficients
    B = _compute_ri_integrals(basis, auxbasis)  # (N_aux, N, N)
    N_aux = B.shape[0]
    n_interp = round(c_isdf * N_aux)
    B_flat = B.reshape(N_aux, N * N)  # (N_aux, N^2)

    # 2. QRCP to select interpolation points
    phi, selected = _isdf_select_points(basis, mesh, n_interp)

    # 3. Collocation matrix
    X = jnp.array(phi[selected, :].T)  # (N, M)

    # 4. Fit Z via RI factored form
    # coll[ij, P] = X[i,P] * X[j,P], shape (N^2, M)
    coll = (X[:, None, :] * X[None, :, :]).reshape(N * N, n_interp)
    coll_np = np.asarray(coll)
    coll_pinv = np.linalg.pinv(coll_np)  # (M, N^2)

    # W = coll_pinv @ B_flat.T  -> (M, N_aux)
    W = coll_pinv @ B_flat.T
    # Z = W @ W.T  -> (M, M)
    Z = jnp.array(W @ W.T)

    return TwoElectronTHC(X=X, Z=Z)
