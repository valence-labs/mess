import jax
import pytest

from mess.basis import basisset
from mess.hamiltonian import Hamiltonian, minimise
from mess.zeropad_integrals import overlap_basis_zeropad
from mess.integrals import (
    eri_basis_sparse,
    kinetic_basis,
    nuclear_basis,
    overlap_basis,
)
from mess.structure import molecule
from mess.interop import from_pyquante
from mess.package_utils import has_package
from conftest import is_mem_limited


@pytest.mark.parametrize("func", [overlap_basis, overlap_basis_zeropad, kinetic_basis])
@pytest.mark.skipif(
    not has_package("pyquante2"), reason="Missing Optional Dependency: pyquante2"
)
def test_benzene(func, benchmark):
    mol = from_pyquante("c6h6")
    basis = basisset(mol, "def2-TZVPPD")
    basis = jax.device_put(basis)

    def harness():
        return func(basis).block_until_ready()

    benchmark(harness)


@pytest.mark.parametrize("mol_name", ["h2", "water"])
@pytest.mark.parametrize(
    "func", [overlap_basis, kinetic_basis, nuclear_basis, eri_basis_sparse]
)
def test_integrals(mol_name, func, benchmark):
    mol = molecule(mol_name)
    basis = basisset(mol)
    basis = jax.device_put(basis)

    def harness():
        return func(basis).block_until_ready()

    benchmark(harness)


@pytest.mark.parametrize("mol_name", ["h2", "water"])
@pytest.mark.skipif(is_mem_limited(), reason="Not enough host memory!")
def test_minimise_ks(benchmark, mol_name):
    # TODO: investigate test failure with cpu backend and float32
    from jax.experimental import enable_x64

    with enable_x64(True):
        mol = molecule(mol_name)
        basis = basisset(mol, "6-31g")
        H = Hamiltonian(basis)
        H = jax.device_put(H)

        def harness():
            E, C, _ = minimise(H)
            return E.block_until_ready(), C.block_until_ready()

        benchmark(harness)
