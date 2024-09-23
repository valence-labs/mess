"""Conversion between Bohr and Angstrom units

Note:
    MESS uses atomic units internally so these conversions are only necessary when
    working with external packages.
"""

from jaxtyping import Array

# Maximum value an individual component of the angular momentum lmn can take
# Used for static ahead-of-time compilation of functions involving lmn.
LMAX = 4

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903


def to_angstrom(bohr_value: Array) -> Array:
    return bohr_value / BOHR_PER_ANGSTROM


def to_bohr(angstrom_value: Array) -> Array:
    return angstrom_value * BOHR_PER_ANGSTROM
