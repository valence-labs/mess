from functools import partial

import numpy as np
import pytest
from jax.experimental import enable_x64
from numpy.testing import assert_allclose
from pyscf import dft

from mess.basis import basisset
from mess.mesh import density_and_grad, xcmesh_from_pyscf
from mess.structure import Structure
from mess.xcfunctional import (
    gga_correlation_lyp,
    gga_correlation_pbe,
    gga_exchange_b88,
    gga_exchange_pbe,
    lda_correlation_pw,
    lda_correlation_vwn,
    lda_exchange,
)

# Define test cases with a mapping from test identifier to test arguments
lda_cases = {
    "lda_exchange": (lda_exchange, "slater,"),
    "lda_correlation_vwn5": (partial(lda_correlation_vwn, use_rpa=False), ",vwn5"),
    "lda_correlation_vwn_rpa": (partial(lda_correlation_vwn, use_rpa=True), ",vwn_rpa"),
    "lda_correlation_pw": (lda_correlation_pw, ",lda_c_pw"),
}

gga_cases = {
    "gga_exchange_b88": (gga_exchange_b88, "gga_x_b88,"),
    "gga_exchange_pbe": (gga_exchange_pbe, "gga_x_pbe,"),
    "gga_correlation_pbe": (gga_correlation_pbe, ",gga_c_pbe"),
    "gga_correlation_lyp": (gga_correlation_lyp, ",gga_c_lyp"),
}


@pytest.fixture
def helium_density():
    with enable_x64(True):
        mol = Structure(np.asarray(2), np.zeros(3))
        basis_name = "6-31g"
        basis = basisset(mol, basis_name)
        mesh = xcmesh_from_pyscf(mol)
        rho, grad_rho = [np.asarray(t) for t in density_and_grad(basis, mesh)]
        yield rho, grad_rho


@pytest.mark.parametrize("xcfunc,scfstr", lda_cases.values(), ids=lda_cases.keys())
def test_lda(helium_density, xcfunc, scfstr):
    rho, _ = helium_density
    actual = xcfunc(rho)
    expect = dft.libxc.eval_xc(scfstr, rho)[0]
    assert_allclose(actual, expect, atol=1e-7)


@pytest.mark.parametrize("xcfunc,scfstr", gga_cases.values(), ids=gga_cases.keys())
def test_gga(helium_density, xcfunc, scfstr):
    rho, grad_rho = helium_density
    scfin = np.concatenate([rho[:, None], grad_rho], axis=1).T

    actual = xcfunc(rho, grad_rho)
    expect = dft.libxc.eval_xc(scfstr, scfin, deriv=1)[0]
    assert_allclose(actual, expect, atol=1e-6)
