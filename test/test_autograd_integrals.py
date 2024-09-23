import numpy as np
from numpy.testing import assert_allclose

from mess.basis import basisset
from mess.interop import to_pyscf
from mess.autograd_integrals import (
    grad_overlap_basis,
    grad_kinetic_basis,
    grad_nuclear_basis,
)
from mess.structure import molecule


def test_nuclear_gradients():
    basis_name = "sto-3g"
    h2 = molecule("h2")
    scfmol = to_pyscf(h2, basis_name)
    basis = basisset(h2, basis_name)

    actual = grad_overlap_basis(basis)
    expect = scfmol.intor("int1e_ipovlp_cart", comp=3)
    assert_allclose(actual, expect, atol=1e-6)

    actual = grad_kinetic_basis(basis)
    expect = scfmol.intor("int1e_ipkin_cart", comp=3)
    assert_allclose(actual, expect, atol=1e-6)

    # TODO: investigate possible inconsistency in libcint outputs?
    actual = grad_nuclear_basis(basis)
    expect = scfmol.intor("int1e_ipnuc_cart", comp=3)
    expect = -np.moveaxis(expect, 1, 2)
    assert_allclose(actual, expect, atol=1e-5)
