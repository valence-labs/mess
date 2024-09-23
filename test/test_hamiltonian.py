import numpy as np
import pytest
from jax.experimental import enable_x64
from numpy.testing import assert_allclose
from pyscf import dft

from mess.basis import basisset
from mess.hamiltonian import Hamiltonian
from mess.interop import to_pyscf
from mess.structure import Structure

cases = {
    "hfx": "hf,",
    "lda": "slater,vwn_rpa",
    "pbe": "gga_x_pbe,gga_c_pbe",
    "pbe0": "pbe0",
    "b3lyp": "b3lyp",
}


@pytest.mark.parametrize("inputs", cases.items(), ids=cases.keys())
def test_energy(inputs):
    with enable_x64(True):
        xc_method, scfxc = inputs
        mol = Structure(np.asarray(2), np.zeros(3))
        basis_name = "6-31g"
        basis = basisset(mol, basis_name)
        scfmol = to_pyscf(mol, basis_name=basis_name)
        s = dft.RKS(scfmol, xc=scfxc)
        s.kernel()
        P = np.asarray(s.make_rdm1())

        H = Hamiltonian(basis=basis, xc_method=xc_method)
        actual = H(P)
        expect = s.energy_tot()
        assert_allclose(actual, expect, atol=1e-6)
