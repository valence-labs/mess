import numpy as np
import pytest
from jax.experimental import enable_x64
from numpy.testing import assert_allclose
from pyscf import dft

from mess.basis import basisset
from mess.hamiltonian import Hamiltonian
from mess.interop import to_pyscf
from mess.mesh import xcmesh_from_pyscf
from mess.structure import Structure, molecule, nuclear_energy
from mess.two_electron import (
    TwoElectronRI,
    TwoElectronTHC,
    isdf_thc,
    isdf_thc_ri,
    ri_from_basis,
)
from mess.units import to_bohr


@pytest.fixture
def water_sto3g():
    with enable_x64(True):
        mol = molecule("water")
        basis = basisset(mol, "sto-3g")
        H = Hamiltonian(basis=basis, xc_method="hfx")

        scfmol = to_pyscf(mol, basis_name="sto-3g")
        s = dft.RKS(scfmol, xc="hf,")
        s.kernel()
        P = np.asarray(s.make_rdm1())

        return H, P, mol, s


@pytest.fixture
def formaldehyde_def2svp():
    with enable_x64(True):
        mol = Structure(
            atomic_number=np.array([6, 8, 1, 1]),
            position=to_bohr(np.array([
                [0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 1.2030],
                [0.0000, 0.9437, -0.5876],
                [0.0000, -0.9437, -0.5876],
            ])),
        )
        basis = basisset(mol, "def2-SVP")
        H = Hamiltonian(basis=basis, xc_method="hfx")

        scfmol = to_pyscf(mol, basis_name="def2-SVP")
        s = dft.RKS(scfmol, xc="hf,")
        s.kernel()
        P = np.asarray(s.make_rdm1())

        return H, P, mol, s


# ---------------------------------------------------------------------------
# RI shape
# ---------------------------------------------------------------------------

def test_ri_shape(water_sto3g):
    """Verify B has dimensions (N_aux, N, N)."""
    with enable_x64(True):
        H, P, mol, _ = water_sto3g
        N = H.basis.num_orbitals
        ri = ri_from_basis(H.basis)
        assert ri.B.ndim == 3
        assert ri.B.shape[1] == N
        assert ri.B.shape[2] == N


# ---------------------------------------------------------------------------
# RI Coulomb
# ---------------------------------------------------------------------------

def test_ri_coulomb_water(water_sto3g):
    """RI-J Coulomb matrix for water/sto-3g vs PySCF, atol=1e-3."""
    with enable_x64(True):
        H, P, mol, scf = water_sto3g
        J_pyscf = np.asarray(scf.get_j(dm=P))

        ri = ri_from_basis(H.basis)
        J_ri = np.asarray(ri.coloumb(P))

        assert_allclose(J_ri, J_pyscf, atol=1e-3)


def test_ri_coulomb_formaldehyde(formaldehyde_def2svp):
    """RI-J Coulomb matrix for formaldehyde/def2-SVP vs PySCF, atol=1e-3."""
    with enable_x64(True):
        H, P, mol, scf = formaldehyde_def2svp
        J_pyscf = np.asarray(scf.get_j(dm=P))

        ri = ri_from_basis(H.basis)
        J_ri = np.asarray(ri.coloumb(P))

        assert_allclose(J_ri, J_pyscf, atol=1e-3)


# ---------------------------------------------------------------------------
# RI Exchange
# ---------------------------------------------------------------------------

def test_ri_exchange_water(water_sto3g):
    """RI-K exchange matrix for water/sto-3g vs PySCF, atol=1e-3."""
    with enable_x64(True):
        H, P, mol, scf = water_sto3g
        K_pyscf = np.asarray(scf.get_k(dm=P))

        ri = ri_from_basis(H.basis)
        K_ri = np.asarray(ri.exchange(P))

        assert_allclose(K_ri, K_pyscf, atol=1e-3)


def test_ri_exchange_formaldehyde(formaldehyde_def2svp):
    """RI-K exchange matrix for formaldehyde/def2-SVP vs PySCF, atol=3e-3."""
    with enable_x64(True):
        H, P, mol, scf = formaldehyde_def2svp
        K_pyscf = np.asarray(scf.get_k(dm=P))

        ri = ri_from_basis(H.basis)
        K_ri = np.asarray(ri.exchange(P))

        assert_allclose(K_ri, K_pyscf, atol=3e-3)


# ---------------------------------------------------------------------------
# RI energy
# ---------------------------------------------------------------------------

def test_ri_energy_water_lda():
    """RI-J total energy for water/sto-3g (LDA) vs PySCF, < 1 mHa."""
    with enable_x64(True):
        mol = molecule("water")
        basis = basisset(mol, "sto-3g")

        scfmol = to_pyscf(mol, basis_name="sto-3g")
        s = dft.RKS(scfmol, xc="slater,vwn_rpa")
        s.kernel()
        P = np.asarray(s.make_rdm1())
        E_pyscf = s.energy_tot()

        ri = ri_from_basis(basis)
        H_ri = Hamiltonian(basis=basis, xc_method="lda", two_electron=ri)
        E_ri = float(H_ri(P)) + float(nuclear_energy(mol))

        assert abs(E_ri - E_pyscf) < 1e-3  # 1 mHa


def test_ri_energy_water_hfx():
    """RI-J/K total energy for water/sto-3g (HFX) vs PySCF, < 1 mHa."""
    with enable_x64(True):
        mol = molecule("water")
        basis = basisset(mol, "sto-3g")

        scfmol = to_pyscf(mol, basis_name="sto-3g")
        s = dft.RKS(scfmol, xc="hf,")
        s.kernel()
        P = np.asarray(s.make_rdm1())
        E_pyscf = s.energy_tot()

        ri = ri_from_basis(basis)
        H_ri = Hamiltonian(basis=basis, xc_method="hfx", two_electron=ri)
        E_ri = float(H_ri(P)) + float(nuclear_energy(mol))

        assert abs(E_ri - E_pyscf) < 1e-3  # 1 mHa


# ---------------------------------------------------------------------------
# THC-from-ERI shape
# ---------------------------------------------------------------------------

def test_thc_shape(water_sto3g):
    """Verify X and Z have the expected dimensions."""
    with enable_x64(True):
        H, P, mol, _ = water_sto3g
        N = H.basis.num_orbitals
        n_interp = 20 * N

        mesh = xcmesh_from_pyscf(H.basis.structure)
        thc = isdf_thc(H.basis, mesh, H.two_electron.eri)

        assert thc.X.shape == (N, n_interp)
        assert thc.Z.shape == (n_interp, n_interp)


# ---------------------------------------------------------------------------
# THC-from-ERI Coulomb
# ---------------------------------------------------------------------------

def test_thc_coulomb_water(water_sto3g):
    """THC Coulomb matrix for water/sto-3g vs PySCF."""
    with enable_x64(True):
        H, P, mol, scf = water_sto3g
        J_pyscf = np.asarray(scf.get_j(dm=P))

        mesh = xcmesh_from_pyscf(H.basis.structure)
        thc = isdf_thc(H.basis, mesh, H.two_electron.eri)
        J_thc = thc.coloumb(P)

        assert_allclose(np.asarray(J_thc), J_pyscf, atol=1e-6)


def test_thc_coulomb_formaldehyde(formaldehyde_def2svp):
    """THC Coulomb matrix for formaldehyde/def2-SVP vs PySCF."""
    with enable_x64(True):
        H, P, mol, scf = formaldehyde_def2svp
        J_pyscf = np.asarray(scf.get_j(dm=P))

        mesh = xcmesh_from_pyscf(H.basis.structure)
        thc = isdf_thc(H.basis, mesh, H.two_electron.eri)
        J_thc = thc.coloumb(P)

        assert_allclose(np.asarray(J_thc), J_pyscf, atol=1e-4)


# ---------------------------------------------------------------------------
# THC-from-ERI Exchange
# ---------------------------------------------------------------------------

def test_thc_exchange_water(water_sto3g):
    """THC exchange matrix for water/sto-3g vs PySCF."""
    with enable_x64(True):
        H, P, mol, scf = water_sto3g
        K_pyscf = np.asarray(scf.get_k(dm=P))

        mesh = xcmesh_from_pyscf(H.basis.structure)
        thc = isdf_thc(H.basis, mesh, H.two_electron.eri)
        K_thc = np.asarray(thc.exchange(P))

        assert_allclose(K_thc, K_pyscf, atol=1e-6)


# ---------------------------------------------------------------------------
# THC-from-ERI energy
# ---------------------------------------------------------------------------

def test_thc_energy_water(water_sto3g):
    """THC total energy for water/sto-3g vs PySCF, sub-microHartree."""
    with enable_x64(True):
        H, P, mol, scf = water_sto3g
        E_pyscf = scf.energy_tot()

        mesh = xcmesh_from_pyscf(H.basis.structure)
        thc = isdf_thc(H.basis, mesh, H.two_electron.eri)
        import equinox as eqx
        H_thc = eqx.tree_at(lambda h: h.two_electron, H, thc)
        E_thc = float(H_thc(P)) + float(nuclear_energy(mol))

        assert abs(E_thc - E_pyscf) < 1e-6


def test_thc_energy_formaldehyde(formaldehyde_def2svp):
    """THC energy for formaldehyde/def2-SVP vs PySCF, < 1 mHartree."""
    with enable_x64(True):
        H, P, mol, scf = formaldehyde_def2svp
        E_pyscf = scf.energy_tot()

        mesh = xcmesh_from_pyscf(H.basis.structure)
        thc = isdf_thc(H.basis, mesh, H.two_electron.eri)
        import equinox as eqx
        H_thc = eqx.tree_at(lambda h: h.two_electron, H, thc)
        E_thc = float(H_thc(P)) + float(nuclear_energy(mol))

        assert abs(E_thc - E_pyscf) < 1e-3


# ---------------------------------------------------------------------------
# THC-from-RI Coulomb
# ---------------------------------------------------------------------------

def test_thc_ri_coulomb_water(water_sto3g):
    """THC-from-RI Coulomb matrix for water/sto-3g vs PySCF, atol=1e-3."""
    with enable_x64(True):
        H, P, mol, scf = water_sto3g
        J_pyscf = np.asarray(scf.get_j(dm=P))

        mesh = xcmesh_from_pyscf(H.basis.structure)
        thc = isdf_thc_ri(H.basis, mesh)
        J_thc = np.asarray(thc.coloumb(P))

        assert_allclose(J_thc, J_pyscf, atol=1e-3)


def test_thc_ri_coulomb_formaldehyde(formaldehyde_def2svp):
    """THC-from-RI Coulomb matrix for formaldehyde/def2-SVP vs PySCF, atol=1e-3."""
    with enable_x64(True):
        H, P, mol, scf = formaldehyde_def2svp
        J_pyscf = np.asarray(scf.get_j(dm=P))

        mesh = xcmesh_from_pyscf(H.basis.structure)
        thc = isdf_thc_ri(H.basis, mesh)
        J_thc = np.asarray(thc.coloumb(P))

        assert_allclose(J_thc, J_pyscf, atol=1e-3)


# ---------------------------------------------------------------------------
# THC-from-RI Exchange
# ---------------------------------------------------------------------------

def test_thc_ri_exchange_water(water_sto3g):
    """THC-from-RI exchange matrix for water/sto-3g vs PySCF, atol=1e-3."""
    with enable_x64(True):
        H, P, mol, scf = water_sto3g
        K_pyscf = np.asarray(scf.get_k(dm=P))

        mesh = xcmesh_from_pyscf(H.basis.structure)
        thc = isdf_thc_ri(H.basis, mesh)
        K_thc = np.asarray(thc.exchange(P))

        assert_allclose(K_thc, K_pyscf, atol=1e-3)


# ---------------------------------------------------------------------------
# THC-from-RI energy
# ---------------------------------------------------------------------------

def test_thc_ri_energy_water(water_sto3g):
    """THC-from-RI total energy for water/sto-3g vs PySCF, < 1 mHa."""
    with enable_x64(True):
        H, P, mol, scf = water_sto3g
        E_pyscf = scf.energy_tot()

        mesh = xcmesh_from_pyscf(H.basis.structure)
        thc = isdf_thc_ri(H.basis, mesh)
        import equinox as eqx
        H_thc = eqx.tree_at(lambda h: h.two_electron, H, thc)
        E_thc = float(H_thc(P)) + float(nuclear_energy(mol))

        assert abs(E_thc - E_pyscf) < 1e-3  # 1 mHa


# ---------------------------------------------------------------------------
# Hamiltonian integration: coulomb= parameter
# ---------------------------------------------------------------------------

def test_hamiltonian_coulomb_ri():
    """Hamiltonian with coulomb='ri' + LDA energy vs PySCF, < 1 mHa."""
    with enable_x64(True):
        mol = molecule("water")
        basis = basisset(mol, "sto-3g")

        scfmol = to_pyscf(mol, basis_name="sto-3g")
        s = dft.RKS(scfmol, xc="slater,vwn_rpa")
        s.kernel()
        P = np.asarray(s.make_rdm1())
        E_pyscf = s.energy_tot()

        H_ri = Hamiltonian(basis=basis, xc_method="lda", coulomb="ri")
        E_ri = float(H_ri(P)) + float(nuclear_energy(mol))

        assert abs(E_ri - E_pyscf) < 1e-3


def test_hamiltonian_coulomb_ri_hfx():
    """Hamiltonian with coulomb='ri' + HFX energy vs PySCF, < 1 mHa."""
    with enable_x64(True):
        mol = molecule("water")
        basis = basisset(mol, "sto-3g")

        scfmol = to_pyscf(mol, basis_name="sto-3g")
        s = dft.RKS(scfmol, xc="hf,")
        s.kernel()
        P = np.asarray(s.make_rdm1())
        E_pyscf = s.energy_tot()

        H_ri = Hamiltonian(basis=basis, xc_method="hfx", coulomb="ri")
        E_ri = float(H_ri(P)) + float(nuclear_energy(mol))

        assert abs(E_ri - E_pyscf) < 1e-3
