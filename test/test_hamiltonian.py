import numpy as np
import pytest
import equinox as eqx
import jax
from jax.experimental import enable_x64
from numpy.testing import assert_allclose
from pyscf import dft

from mess.basis import basisset
from mess.hamiltonian import Hamiltonian
from mess.interop import to_pyscf
from mess.structure import Structure, molecule, nuclear_energy

cases = {
    "hfx": "hf,",
    "lda": "slater,vwn_rpa",
    "pbe": "gga_x_pbe,gga_c_pbe",
    "pbe0": "pbe0",
    "b3lyp": "b3lyp",
}


mol_cases = {
    "water": molecule("water"),
    "He": Structure(np.asarray(2), np.zeros(3)),
}


@pytest.mark.parametrize("inputs", cases.items(), ids=cases.keys())
@pytest.mark.parametrize("basis_name", ["6-31g", "def2-SVP"])
@pytest.mark.parametrize("mol", mol_cases.values(), ids=mol_cases.keys())
def test_energy(inputs, basis_name, mol):
    with enable_x64(True):
        xc_method, scfxc = inputs
        basis = basisset(mol, basis_name)
        scfmol = to_pyscf(mol, basis_name=basis_name)
        s = dft.RKS(scfmol, xc=scfxc)
        s.kernel()
        P = np.asarray(s.make_rdm1())

        H = Hamiltonian(basis=basis, xc_method=xc_method)
        actual = H(P) + nuclear_energy(mol)
        expect = s.energy_tot()
        assert_allclose(actual, expect, atol=1e-6)


def test_autograd_wrt_positions():
    mol = molecule("h2")
    scfmol = to_pyscf(mol, basis_name="def2-SVP")
    s = dft.RKS(scfmol, xc=cases["lda"])
    s.kernel()
    P = np.asarray(s.make_rdm1())
    g = s.Gradients()
    scf_grad = g.kernel()

    @jax.jit
    def f(pos, rest, basis):
        structure = eqx.combine(pos, rest)
        basis = eqx.tree_at(lambda x: x.structure, basis, structure)
        pcenter = structure.position[basis.primitives.atom_index]
        basis = eqx.tree_at(lambda x: x.primitives.center, basis, pcenter)
        H = Hamiltonian(basis=basis, xc_method="lda", backend="mess")

        return H(P) + nuclear_energy(structure)

    mol = jax.device_put(mol)
    basis = basisset(mol, "def2-SVP")
    pos, rest = eqx.partition(mol, lambda x: id(x) == id(mol.position))
    grad_E = jax.grad(f)(pos, rest, basis)
    assert_allclose(-grad_E.position, scf_grad, atol=1e-1)
