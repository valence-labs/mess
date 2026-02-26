"""Two-electron integral representations: full, RI, THC-ISDF, and hybrid RI-J/THC-K.

Provides four factorizations with both Coulomb (J) and exchange (K) builds:

- TwoElectron: full O(N^4) ERI tensor
- TwoElectronRI: Resolution-of-Identity, O(N^2 * N_aux)
- TwoElectronTHC: Tensor Hypercontraction via ISDF, O(N^2 * M + N * M^2)
- TwoElectronTHCRI: Hybrid RI Coulomb (exact) + THC Exchange (approximate)

References:
    Lee, Lin, Head-Gordon, JCTC 2020, 16, 243 (arXiv:1911.00470)
    Dong, Hu, Lin, JCTC 2018, 14, 1311 (arXiv:1711.01531)
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jax.scipy.linalg import qr as jax_qr
from scipy.linalg import cholesky, solve_triangular

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

    def exchange(self, P: FloatNxN, C_occ=None) -> FloatNxN:
        """Build the exchange matrix from the density matrix.

        Args:
            P: the density matrix
            C_occ: ignored, accepted for interface compatibility

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

    def exchange(self, P: FloatNxN, C_occ=None) -> FloatNxN:
        """Build the exchange matrix from the density matrix using RI factors.

        Args:
            P: density matrix (N, N)
            C_occ: ignored, accepted for interface compatibility

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

    def exchange(self, P: FloatNxN, C_occ=None) -> FloatNxN:
        """Build the exchange matrix from the density matrix using THC factors.

        When C_occ is provided, uses the MO form (Eq. 38 of arXiv:1911.00470)
        which costs O(N_occ*NM + N_occ*M^2) instead of the AO form's O(N^2*M + NM^2).

        Args:
            P: density matrix (N, N)
            C_occ: occupied MO coefficients (N, N_occ). If provided, uses
                the cheaper MO-form exchange build.

        Returns:
            Exchange matrix K (N, N)
        """
        if C_occ is not None:
            psi = jnp.einsum("ui,uP->iP", C_occ, self.X)   # (N_occ, M)
            G = 2.0 * jnp.einsum("iP,iQ->PQ", psi, psi)    # (M, M)
        else:
            G = jnp.einsum("iP,ij,jQ->PQ", self.X, P, self.X)  # (M, M)
        return jnp.einsum("kP,PQ,lQ->kl", self.X, self.Z * G, self.X)


class TwoElectronTHCRI(eqx.Module):
    """Hybrid RI-J / THC-K two-electron integrals.

    Uses exact RI Coulomb (via B) and approximate THC exchange (via X, Z).
    This avoids the THC Coulomb error while keeping cheap THC exchange.

    Stores B (N_aux, N, N), collocation X (N, M), and core Z (M, M).
    """

    B: Array   # (N_aux, N, N) — RI coefficients
    X: FloatNxM  # (N, M) — ISDF collocation matrix
    Z: Array   # (M, M) — THC core tensor

    def coloumb(self, P: FloatNxN) -> FloatNxN:
        """Build the Coulomb matrix using exact RI factors.

        Args:
            P: density matrix (N, N)

        Returns:
            Coulomb matrix J (N, N)
        """
        c = jnp.einsum("Pmn,mn->P", self.B, P)
        J = jnp.einsum("Pij,P->ij", self.B, c)
        return J

    def exchange(self, P: FloatNxN, C_occ=None) -> FloatNxN:
        """Build the exchange matrix using THC factors.

        When C_occ is provided, uses the MO form (Eq. 38 of arXiv:1911.00470).

        Args:
            P: density matrix (N, N)
            C_occ: occupied MO coefficients (N, N_occ). If provided, uses
                the cheaper MO-form exchange build.

        Returns:
            Exchange matrix K (N, N)
        """
        if C_occ is not None:
            psi = jnp.einsum("ui,uP->iP", C_occ, self.X)   # (N_occ, M)
            G = 2.0 * jnp.einsum("iP,iQ->PQ", psi, psi)    # (M, M)
        else:
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


def _weighted_kmeans(
    points: np.ndarray,
    weights: np.ndarray,
    n_clusters: int,
    n_iter: int = 25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted K-means clustering (Lloyd's algorithm).

    Minimises sum_mu sum_{r in C_mu} w(r) * ||r - r_mu||^2.

    Args:
        points: data points (G, d)
        weights: positive weights per point (G,)
        n_clusters: number of clusters
        n_iter: maximum Lloyd iterations
        seed: random seed for reproducible initialisation

    Returns:
        centroids: cluster centres (n_clusters, d)
        labels: cluster assignment for each point (G,)
    """
    G, d = points.shape
    rng = np.random.default_rng(seed)

    # Initialise centroids by sampling proportional to weights
    probs = weights / weights.sum()
    idx = rng.choice(G, size=n_clusters, replace=False, p=probs)
    centroids = points[idx].copy()

    pts_sq = np.sum(points ** 2, axis=1)  # (G,) — reused every iteration
    chunk = 10_000  # process grid in chunks to limit memory
    labels = np.empty(G, dtype=np.int32)

    for _ in range(n_iter):
        # --- Assignment: nearest centroid (chunked to cap memory) ---
        cen_sq = np.sum(centroids ** 2, axis=1)  # (K,)
        for s in range(0, G, chunk):
            e = min(s + chunk, G)
            dists = pts_sq[s:e, None] + cen_sq[None, :] - 2.0 * (points[s:e] @ centroids.T)
            labels[s:e] = dists.argmin(axis=1)

        # --- Update: weighted centroid ---
        new_centroids = np.zeros_like(centroids)
        for dim in range(d):
            new_centroids[:, dim] = np.bincount(
                labels, weights=weights * points[:, dim], minlength=n_clusters
            )[:n_clusters]
        total_w = np.bincount(labels, weights=weights, minlength=n_clusters)[:n_clusters]

        # Handle empty clusters by reinitialising from high-weight points
        empty = total_w < 1e-30
        if empty.any():
            new_centroids[empty] = points[rng.choice(G, size=int(empty.sum()), p=probs)]
            total_w[empty] = 1.0

        new_centroids /= total_w[:, None]

        if np.max(np.abs(new_centroids - centroids)) < 1e-10:
            break
        centroids = new_centroids

    return centroids, labels


def _isdf_select_points_qrcp(
    basis: Basis, mesh: Mesh, n_interp: int
) -> tuple[Array, Array]:
    """Evaluate AOs on mesh and select interpolation points via QRCP.

    For small grids, forms pair products zeta (G, N^2) and uses QRCP directly.
    For large grids (zeta > 2 GB), first uses CVT (weighted K-means) to
    spatially prescreen grid points to a manageable candidate set, then
    runs QRCP on the candidates.

    Args:
        basis: the Basis to evaluate AOs
        mesh: quadrature mesh providing grid points
        n_interp: number of interpolation points

    Returns:
        phi: AO values on grid (G, N)
        selected: indices of selected grid points (n_interp,)
    """
    N = basis.num_orbitals
    phi = basis(mesh.points)  # (G, N)
    G = phi.shape[0]

    zeta_bytes = G * N * N * 8
    max_candidates = min(G, max(5 * n_interp, 5000))

    if zeta_bytes < 2e9 or G <= max_candidates:
        # Small/medium grid: full QRCP
        zeta = (phi[:, :, None] * phi[:, None, :]).reshape(G, N * N)
        _, _, piv = jax_qr(zeta.T, pivoting=True)
        selected = piv[:n_interp]
    else:
        # Large grid: CVT spatial prescreening, then QRCP on candidates
        weights = jnp.sum(phi * phi, axis=1)  # (G,)
        grid_points = np.asarray(mesh.points)  # (G, 3)

        _, labels = _weighted_kmeans(grid_points, np.asarray(weights), max_candidates)

        # Select highest-weight grid point from each cluster
        candidates = np.empty(max_candidates, dtype=int)
        for k in range(max_candidates):
            cluster_idx = np.where(labels == k)[0]
            if len(cluster_idx) > 0:
                candidates[k] = cluster_idx[np.argmax(weights[cluster_idx])]
            else:
                candidates[k] = np.argmax(weights)

        # QRCP on candidate subset
        phi_sub = phi[candidates]
        zeta = (phi_sub[:, :, None] * phi_sub[:, None, :]).reshape(max_candidates, N * N)
        _, _, piv = jax_qr(zeta.T, pivoting=True)
        selected = candidates[piv[:n_interp]]

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

    phi, selected = _isdf_select_points_qrcp(basis, mesh, n_interp)

    # Collocation matrix
    X = jnp.array(phi[selected, :].T)  # (N, M)

    # Fit Z: B[ij, P] = X[i,P] * X[j,P], then Z = pinv(B) @ eri_mat @ pinv(B).T
    B = (X[:, None, :] * X[None, :, :]).reshape(N * N, n_interp)  # (N^2, M)
    B_pinv = jnp.linalg.pinv(B)  # (M, N^2)
    eri_mat = eri.reshape(N * N, N * N)
    Z = B_pinv @ eri_mat @ B_pinv.T  # (M, M)

    return TwoElectronTHC(X=X, Z=Z)


def isdf_thc_ri(
    basis: Basis,
    mesh: Mesh,
    auxbasis: str | None = None,
    c_isdf: float = 5.0,
    max_interp_ratio: int = 20,
) -> TwoElectronTHCRI:
    """Fit hybrid RI-J/THC-K factors from RI integrals, never forming the full ERI.

    Returns a TwoElectronTHCRI that uses exact RI Coulomb and approximate THC
    exchange.  Following Lee, Lin, Head-Gordon (JCTC 2020, 16, 243), the number
    of interpolation points is N_IP = c_ISDF * N_X, where N_X is the auxiliary
    basis size.  Interpolation points are selected via CVT (weighted K-means)
    following Dong, Hu, Lin (JCTC 2018, 14, 1311).

    Args:
        basis: the Basis to evaluate AOs
        mesh: quadrature mesh providing grid points
        auxbasis: auxiliary basis set name (None = PySCF auto-selects)
        c_isdf: ISDF interpolation ratio; N_IP = c_ISDF * N_aux (default: 5.0)
        max_interp_ratio: cap N_IP at max_interp_ratio * N (default: 20)

    Returns:
        TwoElectronTHCRI with RI coefficients B, collocation X, and core Z
    """
    N = basis.num_orbitals

    # 1. Compute RI coefficients
    B = _compute_ri_integrals(basis, auxbasis)  # (N_aux, N, N)
    N_aux = B.shape[0]
    n_interp = min(round(c_isdf * N_aux), max_interp_ratio * N)

    # 2. Select interpolation points (CVT prescreening + QRCP for large grids)
    phi, selected = _isdf_select_points_qrcp(basis, mesh, n_interp)

    # 3. Collocation matrix
    X = jnp.array(phi[selected, :].T)  # (N, M)

    # 4. Fit Z via RI factored form
    # coll[ij, P] = X[i,P] * X[j,P], shape (N^2, M)
    coll = (X[:, None, :] * X[None, :, :]).reshape(N * N, n_interp)
    coll_pinv = jnp.linalg.pinv(coll)  # (M, N^2)

    B_flat = jnp.array(B).reshape(N_aux, N * N)  # (N_aux, N^2)
    W = coll_pinv @ B_flat.T  # (M, N_aux)
    Z = W @ W.T  # (M, M)

    return TwoElectronTHCRI(B=jnp.array(B), X=X, Z=Z)
