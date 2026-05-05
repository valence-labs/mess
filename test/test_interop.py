import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from mess.basis import basisset, renorm
from mess.interop import to_pyscf
from mess.mesh import density, density_and_grad, uniform_mesh
from mess.structure import molecule, nuclear_energy, Structure
from conftest import is_mem_limited

mol_cases = {
    "water": molecule("water"),
    "Kr": Structure(np.asarray(36), np.zeros(3)),
}


@pytest.mark.parametrize("basis_name", ["sto-3g", "6-31g**"])
@pytest.mark.parametrize("spherical", [True, False])
def test_to_pyscf(basis_name, spherical):
    mol = molecule("water")
    basis = basisset(mol, basis_name, spherical)
    pyscf_mol = to_pyscf(mol, basis_name, spherical)
    assert basis.num_orbitals == pyscf_mol.nao


@pytest.mark.skipif(is_mem_limited(), reason="Not enough host memory!")
@pytest.mark.parametrize("basis_name", ["6-31g*", "def2-SVP"])
@pytest.mark.parametrize("spherical", [True, False])
@pytest.mark.parametrize("mol", mol_cases.values(), ids=mol_cases.keys())
def test_gto(basis_name, spherical, mol):
    from pyscf.dft.numint import eval_rho, eval_ao
    from jax import enable_x64

    with enable_x64(True):
        # Run these comparisons to PySCF in fp64
        # Atomic orbitals
        basis = basisset(mol, basis_name, spherical)
        basis = renorm(basis, mode="pyscf_sph" if spherical else "pyscf_cart")
        mesh = uniform_mesh()
        actual = basis(mesh.points)

        mol = to_pyscf(mol, basis_name, spherical)
        expect_ao = eval_ao(mol, np.asarray(mesh.points))
        assert_allclose(actual, expect_ao, atol=1e-7)

        # Density Matrix
        mf = mol.RKS()
        mf.kernel()
        C = jnp.array(mf.mo_coeff)
        P = basis.density_matrix(C)
        expect = jnp.array(mf.make_rdm1())
        assert_allclose(P, expect, atol=1e-7)

        # Electron density
        actual = density(basis, mesh, P)
        expect = eval_rho(mol, expect_ao, mf.make_rdm1(), xctype="lda")
        assert_allclose(actual, expect, atol=1e-6)

        # Electron density and gradient
        rho, grad_rho = density_and_grad(basis, mesh, P)
        ao_and_grad = eval_ao(mol, np.asarray(mesh.points), deriv=1)
        expect = eval_rho(mol, ao_and_grad, mf.make_rdm1(), xctype="gga")
        expect_rho = expect[0, :]
        expect_grad = expect[1:, :].T
        assert_allclose(rho, expect_rho, atol=1e-6)
        assert_allclose(grad_rho, expect_grad, atol=1e-5)


@pytest.mark.parametrize("name", ["water", "h2"])
def test_nuclear_energy(name):
    mol = molecule(name)
    actual = nuclear_energy(mol)
    expect = to_pyscf(mol).energy_nuc()
    assert_allclose(actual, expect)
