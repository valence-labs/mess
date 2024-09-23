"""Interoperation tools for working across MESS, PySCF, and PyQuante."""

from typing import Tuple

import numpy as np
from periodictable import elements
from pyscf import gto
from pyquante2.geo import samples

from mess.basis import Basis, basisset
from mess.structure import Structure
from mess.units import to_bohr


def to_pyscf(structure: Structure, basis_name: str = "sto-3g") -> "gto.Mole":
    mol = gto.Mole(unit="Bohr", spin=structure.num_electrons % 2, cart=True)
    mol.atom = [
        (symbol, pos)
        for symbol, pos in zip(structure.atomic_symbol, structure.position)
    ]
    mol.basis = basis_name
    mol.build(unit="Bohr")
    return mol


def from_pyscf(mol: "gto.Mole") -> Tuple[Structure, Basis]:
    atoms = [(elements.symbol(sym).number, pos) for sym, pos in mol.atom]
    atomic_number, position = [np.array(x) for x in zip(*atoms)]

    if mol.unit == "Angstrom":
        position = to_bohr(position)

    structure = Structure(atomic_number, position)

    basis = basisset(structure, basis_name=mol.basis)

    return structure, basis


def from_pyquante(name: str) -> Structure:
    """Load molecular structure from pyquante2.geo.samples module

    Args:
        name (str): Possible names include ch4, c6h6, aspirin, caffeine, hmx, petn,
                    prozan, rdx, taxol, tylenol, viagara, zoloft

    Returns:
        Structure
    """
    pqmol = getattr(samples, name)
    atomic_number, position = zip(*[(a.Z, a.r) for a in pqmol])
    atomic_number, position = [np.asarray(x) for x in (atomic_number, position)]
    return Structure(atomic_number, position)
