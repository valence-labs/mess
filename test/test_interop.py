import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from mess.basis import basisset
from mess.interop import to_pyscf
from mess.mesh import density, density_and_grad, uniform_mesh
from mess.structure import molecule, nuclear_energy


@pytest.mark.parametrize("basis_name", ["sto-3g", "6-31g**"])
def test_to_pyscf(basis_name):
    mol = molecule("water")
    basis = basisset(mol, basis_name)
    pyscf_mol = to_pyscf(mol, basis_name)
    assert basis.num_orbitals == pyscf_mol.nao


def test_gto():
    from pyscf.dft.numint import eval_rho, eval_ao
    from jax.experimental import enable_x64

    with enable_x64(True):
        # Run these comparisons to PySCF in fp64
        # Atomic orbitals
        basis_name = "6-31+g"
        structure = molecule("water")
        basis = basisset(structure, basis_name)
        mesh = uniform_mesh()
        actual = basis(mesh.points)

        mol = to_pyscf(structure, basis_name)
        expect_ao = eval_ao(mol, np.asarray(mesh.points))
        assert_allclose(actual, expect_ao, atol=1e-7)

        # Density Matrix
        mf = mol.RKS()
        mf.kernel()
        C = jnp.array(mf.mo_coeff)
        P = basis.density_matrix(C)
        expect = jnp.array(mf.make_rdm1())
        assert_allclose(P, expect)

        # Electron density
        actual = density(basis, mesh, P)
        expect = eval_rho(mol, expect_ao, mf.make_rdm1(), xctype="lda")
        assert_allclose(actual, expect, atol=1e-7)

        # Electron density and gradient
        rho, grad_rho = density_and_grad(basis, mesh, P)
        ao_and_grad = eval_ao(mol, np.asarray(mesh.points), deriv=1)
        expect = eval_rho(mol, ao_and_grad, mf.make_rdm1(), xctype="gga")
        expect_rho = expect[0, :]
        expect_grad = expect[1:, :].T
        assert_allclose(rho, expect_rho, atol=1e-7)
        assert_allclose(grad_rho, expect_grad, atol=1e-6)


@pytest.mark.parametrize("name", ["water", "h2"])
def test_nuclear_energy(name):
    mol = molecule(name)
    actual = nuclear_energy(mol)
    expect = to_pyscf(mol).energy_nuc()
    assert_allclose(actual, expect)
