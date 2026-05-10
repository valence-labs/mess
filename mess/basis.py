"""basis sets of Gaussian type orbitals"""

from functools import cache
from typing import Literal, Tuple, get_args

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import jit
from jax.ops import segment_sum

from mess.orbital import Orbital, batch_orbitals
from mess.primitive import Primitive
from mess.shell import make_shell
from mess.structure import Structure
from mess.types import (
    FloatN,
    FloatNx3,
    FloatNxM,
    FloatNxN,
    IntN,
    default_fptype,
)


class Basis(eqx.Module):
    orbitals: Tuple[Orbital]
    structure: Structure
    primitives: Primitive
    coefficients: FloatN
    orbital_index: IntN
    basis_name: str = eqx.field(static=True)
    max_L: int = eqx.field(static=True)
    spherical: bool = eqx.field(static=True)

    @property
    def num_orbitals(self) -> int:
        return len(self.orbitals)

    @property
    def num_primitives(self) -> int:
        return sum(ao.num_primitives for ao in self.orbitals)

    @property
    def occupancy(self) -> FloatN:
        """Restricted occupancy: 2 electrons per orbital until filled."""
        occ = jnp.full(self.num_orbitals, 2.0)
        mask = occ.cumsum() > self.structure.num_electrons
        occ = jnp.where(mask, 0.0, occ)
        return occ

    @property
    def occupancy_alpha(self) -> FloatN:
        """Alpha spin occupancy: 1 electron per orbital until n_alpha filled."""
        occ = jnp.full(self.num_orbitals, 1.0)
        mask = occ.cumsum() > self.structure.n_alpha
        occ = jnp.where(mask, 0.0, occ)
        return occ

    @property
    def occupancy_beta(self) -> FloatN:
        """Beta spin occupancy: 1 electron per orbital until n_beta filled."""
        occ = jnp.full(self.num_orbitals, 1.0)
        mask = occ.cumsum() > self.structure.n_beta
        occ = jnp.where(mask, 0.0, occ)
        return occ

    def to_dataframe(self) -> pd.DataFrame:
        def fixer(x):
            # simple workaround for storing 2d array as a pandas column
            return [x[i, :] for i in range(x.shape[0])]

        df = pd.DataFrame()
        df["orbital"] = self.orbital_index
        df["atom"] = self.primitives.atom_index
        df["coefficient"] = self.coefficients
        df["norm"] = self.primitives.norm
        df["center"] = fixer(self.primitives.center)
        df["lmn"] = fixer(self.primitives.lmn)
        df["alpha"] = self.primitives.alpha
        df.index.name = "primitive"
        return df

    def density_matrix(self, C: FloatNxN) -> FloatNxN:
        """Evaluate the restricted density matrix from MO coefficients.

        Args:
            C (FloatNxN): the molecular orbital coefficients

        Returns:
            FloatNxN: the density matrix.
        """
        return jnp.einsum("k,ik,jk->ij", self.occupancy, C, C)

    def density_matrix_alpha(self, C: FloatNxN) -> FloatNxN:
        """Evaluate the alpha spin density matrix from MO coefficients.

        Args:
            C (FloatNxN): the alpha molecular orbital coefficients

        Returns:
            FloatNxN: the alpha density matrix.
        """
        return jnp.einsum("k,ik,jk->ij", self.occupancy_alpha, C, C)

    def density_matrix_beta(self, C: FloatNxN) -> FloatNxN:
        """Evaluate the beta spin density matrix from MO coefficients.

        Args:
            C (FloatNxN): the beta molecular orbital coefficients

        Returns:
            FloatNxN: the beta density matrix.
        """
        return jnp.einsum("k,ik,jk->ij", self.occupancy_beta, C, C)

    @jit
    def __call__(self, pos: FloatNx3) -> FloatNxM:
        prim = self.coefficients[jnp.newaxis, :] * self.primitives(pos)
        orb = segment_sum(prim.T, self.orbital_index, num_segments=self.num_orbitals)
        return orb.T

    def __repr__(self) -> str:
        return repr(self.to_dataframe())

    def _repr_html_(self) -> str | None:
        df = self.to_dataframe()
        return df._repr_html_()

    def __hash__(self) -> int:
        return hash(self.primitives)


def basisset(
    structure: Structure, basis_name: str = "sto-3g", spherical: bool = True
) -> Basis:
    """Factory function for building a basis set for a structure.

    Args:
        structure (Structure): Used to define the basis function parameters.
        basis_name (str, optional): Basis set name to look up on the
            `basis set exchange <https://www.basissetexchange.org/>`_.
            Defaults to ``sto-3g``.
        spherical (bool): flag to enable using spherical format Gaussian basis functions
            as opposed to Cartesian format. Defaults to ``True``.

    Returns:
        Basis constructed from inputs
    """
    orbitals = []
    atom_index = []

    for atom_id in range(structure.num_atoms):
        element = int(structure.atomic_number[atom_id])
        out = _bse_to_orbitals(basis_name, element, spherical)
        atom_index.extend([atom_id] * sum(len(ao.primitives) for ao in out))
        orbitals += out

    primitives, coefficients, orbital_index = batch_orbitals(orbitals)
    primitives = eqx.tree_at(lambda p: p.atom_index, primitives, jnp.array(atom_index))
    center = structure.position[primitives.atom_index, :]
    primitives = eqx.tree_at(lambda p: p.center, primitives, center)

    basis = Basis(
        orbitals=orbitals,
        structure=structure,
        primitives=primitives,
        coefficients=coefficients,
        orbital_index=orbital_index,
        basis_name=basis_name,
        max_L=int(np.max(primitives.lmn)),
        spherical=spherical,
    )

    # TODO(hh): this introduces some performance overhead into basis construction that
    # could be pushed down into the cached orbitals.
    basis = renorm(basis)
    return basis


@cache
def _bse_to_orbitals(
    basis_name: str, atomic_number: int, spherical: bool
) -> Tuple[Orbital]:
    """
    Look up basis set parameters on the basis set exchange and build a tuple of Orbital.

    The output is cached to reuse the same objects for a given basis set and atomic
    number.  This can help save time when batching over different coordinates.

    Args:
        basis_name (str): The name of the basis set to lookup on the basis set exchange.
        atomic_number (int): The atomic number for the element to retrieve.
        spherical (bool): flag to enable using spherical format Gaussian basis functions
            as opposed to Cartesian format.

    Returns:
        Tuple[Orbital]: Tuple of Orbital objects corresponding to the specified basis
            set and atomic number.
    """
    from basis_set_exchange import get_basis
    from basis_set_exchange.sort import sort_basis

    bse_basis = get_basis(
        basis_name,
        elements=atomic_number,
        uncontract_spdf=True,
        uncontract_general=True,
    )
    bse_basis = sort_basis(bse_basis)["elements"]
    orbitals = []
    zero_center = np.zeros(3, dtype=default_fptype())

    for s in bse_basis[str(atomic_number)]["electron_shells"]:
        L, alphas, coefficients = _parse_bse_shell(s)
        assert len(coefficients) == len(alphas), "Expecting same size vectors!"
        orbitals += make_shell(spherical, L, zero_center, alphas, coefficients)

    return tuple(orbitals)


def _parse_bse_shell(shell_dict):
    return (
        shell_dict["angular_momentum"][0],
        np.array(shell_dict["exponents"], dtype=default_fptype()).reshape(-1),
        np.array(shell_dict["coefficients"], dtype=default_fptype()).reshape(-1),
    )


def basis_iter(basis: Basis):
    from jax import tree

    from mess.special import triu_indices

    def take_primitives(indices):
        p = tree.map(lambda x: jnp.take(x, indices, axis=0), basis.primitives)
        c = jnp.take(basis.coefficients, indices)
        return p, c

    ii, jj = triu_indices(basis.num_primitives)
    lhs, cl = take_primitives(ii.reshape(-1))
    rhs, cr = take_primitives(jj.reshape(-1))
    return (ii, cl, lhs), (jj, cr, rhs)


RenormMode = Literal["orthonormal", "pyscf_cart", "pyscf_sph"]


def renorm(basis: Basis, mode: RenormMode = "orthonormal") -> Basis:
    """Renormalise the basis set.

    Args:
        basis (Basis): The basis set to renormalise.
        mode (str, optional): The normalisation mode. Can be "orthonormal" to
            normalise such that the self-overlap integral is 1, or either "pyscf_cart"
            or "pyscf_sph" to normalise according to PySCF's convention.
            Defaults to "orthonormal".

    Raises:
        ValueError: If an unknown normalisation mode is provided.

    Returns:
        Basis: The renormalised basis set.
    """
    match mode:
        case "orthonormal":
            from mess.integrals import overlap_basis

            S = overlap_basis(basis)
            n = 1 / jnp.sqrt(jnp.diag(S))
        case "pyscf_cart" | "pyscf_sph":
            from mess.interop import to_pyscf

            mol = to_pyscf(basis.structure, basis.basis_name)
            kind = mode.split("_")[1]
            n = np.sqrt(np.diag(mol.intor(f"int1e_ovlp_{kind}")))
        case _:
            modes = get_args(RenormMode)
            modes = ", ".join(modes)
            msg = f"Unknown renorm mode: {mode}"
            msg += f"\nMust be one of the following: {modes}"
            raise ValueError(msg)

    C = n[basis.orbital_index] * basis.coefficients
    return eqx.tree_at(lambda b: b.coefficients, basis, C)
