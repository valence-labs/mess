"""basis sets of Gaussian type orbitals"""

from functools import cache
from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import jit
from jax.ops import segment_sum

from mess.orbital import Orbital, batch_orbitals
from mess.primitive import Primitive
from mess.structure import Structure
from mess.types import (
    Float3,
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

    @property
    def num_orbitals(self) -> int:
        return len(self.orbitals)

    @property
    def num_primitives(self) -> int:
        return sum(ao.num_primitives for ao in self.orbitals)

    @property
    def occupancy(self) -> FloatN:
        # Assumes uncharged systems in restricted Kohn-Sham
        occ = jnp.full(self.num_orbitals, 2.0)
        mask = occ.cumsum() > self.structure.num_electrons
        occ = jnp.where(mask, 0.0, occ)
        return occ

    def to_dataframe(self) -> pd.DataFrame:
        def fixer(x):
            # simple workaround for storing 2d array as a pandas column
            return [x[i, :] for i in range(x.shape[0])]

        df = pd.DataFrame()
        df["orbital"] = self.orbital_index
        df["coefficient"] = self.coefficients
        df["norm"] = self.primitives.norm
        df["center"] = fixer(self.primitives.center)
        df["lmn"] = fixer(self.primitives.lmn)
        df["alpha"] = self.primitives.alpha
        df.index.name = "primitive"
        return df

    def density_matrix(self, C: FloatNxN) -> FloatNxN:
        """Evaluate the density matrix from the molecular orbital coefficients

        Args:
            C (FloatNxN): the molecular orbital coefficients

        Returns:
            FloatNxN: the density matrix.
        """
        return jnp.einsum("k,ik,jk->ij", self.occupancy, C, C)

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


def basisset(structure: Structure, basis_name: str = "sto-3g") -> Basis:
    """Factory function for building a basis set for a structure.

    Args:
        structure (Structure): Used to define the basis function parameters.
        basis_name (str, optional): Basis set name to look up on the
            `basis set exchange <https://www.basissetexchange.org/>`_.
            Defaults to ``sto-3g``.

    Returns:
        Basis constructed from inputs
    """
    orbitals = []

    for a in range(structure.num_atoms):
        element = int(structure.atomic_number[a])
        center = structure.position[a, :]
        orbitals += _build_orbitals(basis_name, element, center)

    primitives, coefficients, orbital_index = batch_orbitals(orbitals)

    return Basis(
        orbitals=orbitals,
        structure=structure,
        primitives=primitives,
        coefficients=coefficients,
        orbital_index=orbital_index,
        basis_name=basis_name,
        max_L=int(np.max(primitives.lmn)),
    )


@cache
def _bse_to_orbitals(basis_name: str, atomic_number: int) -> Tuple[Orbital]:
    """
    Look up basis set parameters on the basis set exchange and build a tuple of Orbital.

    The output is cached to reuse the same objects for a given basis set and atomic
    number.  This can help save time when batching over different coordinates.

    Args:
        basis_name (str): The name of the basis set to lookup on the basis set exchange.
        atomic_number (int): The atomic number for the element to retrieve.

    Returns:
        Tuple[Orbital]: Tuple of Orbital objects corresponding to the specified basis
            set and atomic number.
    """
    from basis_set_exchange import get_basis
    from basis_set_exchange.sort import sort_basis

    # fmt: off
    LMN_MAP = {
        0: [(0, 0, 0)],
        1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        2: [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)],
        3: [(3, 0, 0), (2, 1, 0), (2, 0, 1), (1, 2, 0), (1, 1, 1),
            (1, 0, 2), (0, 3, 0), (0, 2, 1), (0, 1, 2), (0, 0, 3)],
    }
    # fmt: on

    bse_basis = get_basis(
        basis_name,
        elements=atomic_number,
        uncontract_spdf=True,
        uncontract_general=True,
    )
    bse_basis = sort_basis(bse_basis)["elements"]
    orbitals = []

    for s in bse_basis[str(atomic_number)]["electron_shells"]:
        for lmn in LMN_MAP[s["angular_momentum"][0]]:
            ao = Orbital.from_bse(
                center=np.zeros(3, dtype=default_fptype()),
                alphas=np.array(s["exponents"], dtype=default_fptype()),
                lmn=np.array(lmn, dtype=np.int32),
                coefficients=np.array(s["coefficients"], dtype=default_fptype()),
            )
            orbitals.append(ao)

    return tuple(orbitals)


def _build_orbitals(
    basis_name: str, atomic_number: int, center: Float3
) -> Tuple[Orbital]:
    """
    Constructs a tuple of Orbital objects for a given atomic_number and basis set,
    with each orbital centered at the specified coordinates.

    Args:
        basis_name (str): The name of the basis set to use.
        atomic_number (int): The atomic number used to build the orbitals.
        center (Float3): the 3D coordinate specifying the center of the orbitals

    Returns:
        Tuple[Orbital]: A tuple of Orbitals centered at the provided coordinates.
    """
    orbitals = _bse_to_orbitals(basis_name, atomic_number)

    def where(orbitals):
        return [p.center for ao in orbitals for p in ao.primitives]

    num_centers = len(where(orbitals))
    return eqx.tree_at(where, orbitals, replace=np.tile(center, (num_centers, 1)))


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
