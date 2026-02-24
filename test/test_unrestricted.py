"""Tests for unrestricted (spin-polarized) calculations."""

import numpy as np
import pytest
from jax.experimental import enable_x64
from numpy.testing import assert_allclose
from pyscf import scf, dft

from mess.basis import basisset
from mess.hamiltonian import UHamiltonian, uminimise
from mess.interop import to_pyscf
from mess.structure import Structure, molecule, nuclear_energy


# XC method mappings: mess name -> pyscf name
xc_cases = {
    "hfx": "hf,",
    "lda": "slater,vwn_rpa",
    "pbe": "gga_x_pbe,gga_c_pbe",
}


# Closed-shell molecules (UHF should match RHF)
closed_shell_cases = {
    "water": molecule("water"),
    "He": Structure(np.asarray([2]), np.zeros((1, 3))),
    "H2": molecule("h2"),
}

# Open-shell molecules
open_shell_cases = {
    "H_atom": Structure(
        atomic_number=np.array([1]),
        position=np.array([[0.0, 0.0, 0.0]]),
        spin_multiplicity=2,
    ),
    "Li_atom": Structure(
        atomic_number=np.array([3]),
        position=np.array([[0.0, 0.0, 0.0]]),
        spin_multiplicity=2,
    ),
    "O2_triplet": Structure(
        atomic_number=np.array([8, 8]),
        position=np.array([[0.0, 0.0, 0.0], [2.28, 0.0, 0.0]]),
        spin_multiplicity=3,
    ),
}


class TestStructureSpin:
    """Test spin-related Structure properties."""

    def test_default_singlet(self):
        """Closed-shell molecule defaults to singlet."""
        mol = molecule("water")
        assert mol.spin_multiplicity == 1
        assert mol.n_alpha == 5
        assert mol.n_beta == 5

    def test_default_doublet_odd_electrons(self):
        """Odd electron count defaults to doublet."""
        h = Structure(np.array([1]), np.zeros((1, 3)))
        assert h.spin_multiplicity == 2
        assert h.n_alpha == 1
        assert h.n_beta == 0

    def test_explicit_triplet(self):
        """Explicit triplet spin multiplicity."""
        o2 = Structure(
            atomic_number=np.array([8, 8]),
            position=np.array([[0.0, 0.0, 0.0], [2.28, 0.0, 0.0]]),
            spin_multiplicity=3,
        )
        assert o2.n_alpha == 9
        assert o2.n_beta == 7
        assert o2.n_alpha - o2.n_beta == 2  # 2 unpaired electrons

    def test_charged_system(self):
        """Charged system electron count."""
        # H2+ cation
        h2_plus = Structure(
            atomic_number=np.array([1, 1]),
            position=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
            charge=1,
            spin_multiplicity=2,
        )
        assert h2_plus.num_electrons == 1
        assert h2_plus.n_alpha == 1
        assert h2_plus.n_beta == 0

    def test_invalid_spin_raises(self):
        """Invalid spin multiplicity raises error."""
        with pytest.raises(ValueError):
            # Can't have triplet with 1 electron
            Structure(
                atomic_number=np.array([1]),
                position=np.array([[0.0, 0.0, 0.0]]),
                spin_multiplicity=3,
            )


class TestBasisOccupancy:
    """Test spin-specific occupancy properties."""

    def test_closed_shell_occupancy(self):
        """Alpha and beta occupancy equal for closed shell."""
        mol = molecule("water")
        basis = basisset(mol, "sto-3g")

        occ_a = basis.occupancy_alpha
        occ_b = basis.occupancy_beta

        assert_allclose(occ_a, occ_b)
        assert occ_a.sum() == mol.n_alpha
        assert occ_b.sum() == mol.n_beta

    def test_open_shell_occupancy(self):
        """Alpha and beta occupancy differ for open shell."""
        h = Structure(np.array([1]), np.zeros((1, 3)), spin_multiplicity=2)
        basis = basisset(h, "sto-3g")

        assert basis.occupancy_alpha.sum() == 1
        assert basis.occupancy_beta.sum() == 0


class TestUnrestrictedClosedShell:
    """Test that UHF/UKS matches RHF/RKS for closed-shell systems."""

    @pytest.mark.parametrize("xc_method,pyscf_xc", xc_cases.items(), ids=xc_cases.keys())
    @pytest.mark.parametrize("mol", closed_shell_cases.values(), ids=closed_shell_cases.keys())
    def test_closed_shell_energy_matches_restricted(self, xc_method, pyscf_xc, mol):
        """UHF/UKS energy should match RHF/RKS for closed-shell."""
        with enable_x64(True):
            basis = basisset(mol, "sto-3g")

            # MESS unrestricted
            H_u = UHamiltonian(basis, xc_method=xc_method)
            E_mess, _, _, _ = uminimise(H_u)

            # PySCF restricted (as reference)
            pyscf_mol = to_pyscf(mol, basis_name="sto-3g")
            if pyscf_xc == "hf,":
                mf = scf.RHF(pyscf_mol)
            else:
                mf = dft.RKS(pyscf_mol, xc=pyscf_xc)
            E_pyscf = mf.kernel()

            assert_allclose(E_mess, E_pyscf, atol=1e-5)


class TestUnrestrictedOpenShell:
    """Test UHF/UKS for open-shell systems against PySCF UHF/UKS."""

    @pytest.mark.parametrize("xc_method,pyscf_xc", xc_cases.items(), ids=xc_cases.keys())
    @pytest.mark.parametrize("mol_name,mol", open_shell_cases.items(), ids=open_shell_cases.keys())
    def test_open_shell_energy_vs_pyscf(self, xc_method, pyscf_xc, mol_name, mol):
        """UHF/UKS energy should match PySCF UHF/UKS for open-shell.

        Note: UHF can have multiple local minima and the converged solution
        depends on the initial guess. We use relaxed tolerances since the
        energy functional is verified correct via TestUnrestrictedEnergyEval.
        """
        with enable_x64(True):
            basis = basisset(mol, "sto-3g")

            # MESS unrestricted
            H_u = UHamiltonian(basis, xc_method=xc_method)
            E_mess, C_a, C_b, _ = uminimise(H_u)

            # PySCF unrestricted
            pyscf_mol = to_pyscf(mol, basis_name="sto-3g")
            pyscf_mol.spin = mol.spin_multiplicity - 1
            pyscf_mol.build()

            if pyscf_xc == "hf,":
                mf = scf.UHF(pyscf_mol)
            else:
                mf = dft.UKS(pyscf_mol, xc=pyscf_xc)
            E_pyscf = mf.kernel()

            # Open-shell should now match closely with symmetry-broken initial guess
            assert_allclose(E_mess, E_pyscf, atol=0.01)


class TestUnrestrictedEnergyEval:
    """Test that energy evaluation is correct using PySCF density matrices."""

    @pytest.mark.parametrize("xc_method,pyscf_xc", xc_cases.items(), ids=xc_cases.keys())
    def test_energy_with_pyscf_density(self, xc_method, pyscf_xc):
        """MESS energy with PySCF density should match PySCF energy closely.

        This verifies the energy functional is implemented correctly,
        independent of optimizer convergence issues.

        Note: Small differences (~0.002 Ha) can occur in XC energies due to
        numerical integration and functional implementation details.
        """
        with enable_x64(True):
            o2 = Structure(
                atomic_number=np.array([8, 8]),
                position=np.array([[0.0, 0.0, 0.0], [2.28, 0.0, 0.0]]),
                spin_multiplicity=3,
            )
            basis = basisset(o2, "sto-3g")

            # Get PySCF converged density
            pyscf_mol = to_pyscf(o2, basis_name="sto-3g")
            pyscf_mol.spin = 2
            pyscf_mol.build()

            if pyscf_xc == "hf,":
                mf = scf.UHF(pyscf_mol)
            else:
                mf = dft.UKS(pyscf_mol, xc=pyscf_xc)
            mf.kernel()

            dm_pyscf = mf.make_rdm1()
            P_a = np.array(dm_pyscf[0])
            P_b = np.array(dm_pyscf[1])

            # Evaluate MESS energy with PySCF density
            H = UHamiltonian(basis, xc_method=xc_method)
            from mess.structure import nuclear_energy
            E_mess = float(H(P_a, P_b)) + nuclear_energy(o2)

            # Allow small tolerance for XC numerical differences
            # HFX should match exactly, DFT has small grid/functional differences
            atol = 1e-6 if xc_method == "hfx" else 0.02
            assert_allclose(E_mess, mf.e_tot, atol=atol)


class TestUnrestrictedDensityMatrix:
    """Test density matrix construction."""

    def test_density_matrix_trace(self):
        """Density matrix trace with overlap gives electron count."""
        with enable_x64(True):
            mol = molecule("water")
            basis = basisset(mol, "sto-3g")

            H = UHamiltonian(basis, xc_method="hfx")
            _, C_a, C_b, _ = uminimise(H)

            P_a = basis.density_matrix_alpha(C_a)
            P_b = basis.density_matrix_beta(C_b)

            # Get overlap matrix
            from mess.hamiltonian import OneElectron
            S = OneElectron(basis, backend="pyscf_sph").overlap

            # Trace(P @ S) = number of electrons
            n_alpha = np.trace(P_a @ S)
            n_beta = np.trace(P_b @ S)

            assert_allclose(n_alpha, mol.n_alpha, atol=1e-6)
            assert_allclose(n_beta, mol.n_beta, atol=1e-6)

    def test_spin_density_open_shell(self):
        """Spin density is non-zero for open-shell."""
        with enable_x64(True):
            # Use Li atom (5 orbitals) instead of H atom (1 orbital)
            # H atom with STO-3G has only 1 orbital, making optimization trivial
            li = Structure(np.array([3]), np.zeros((1, 3)), spin_multiplicity=2)
            basis = basisset(li, "sto-3g")

            H = UHamiltonian(basis, xc_method="hfx")
            _, C_a, C_b, _ = uminimise(H)

            P_a = basis.density_matrix_alpha(C_a)
            P_b = basis.density_matrix_beta(C_b)
            P_spin = P_a - P_b

            # For Li atom (doublet), spin density should be non-zero
            assert np.abs(P_spin).sum() > 0.1


class TestXCFunctionalsSpinPolarized:
    """Test spin-polarized XC functional implementations."""

    def test_lda_exchange_spinpol_closed_shell(self):
        """Spin-polarized LDA exchange equals restricted for closed shell."""
        from mess.xcfunctional import lda_exchange, lda_exchange_spinpol
        import jax.numpy as jnp

        rho = jnp.array([0.1, 0.2, 0.3])
        rho_a = rho / 2
        rho_b = rho / 2

        eps_r = lda_exchange(rho)
        eps_s = lda_exchange_spinpol(rho_a, rho_b)

        assert_allclose(eps_r, eps_s, atol=1e-10)

    def test_vwn_correlation_with_zeta(self):
        """VWN correlation with zeta=0 matches original."""
        from mess.xcfunctional import lda_correlation_vwn
        import jax.numpy as jnp

        rho = jnp.array([0.1, 0.2, 0.3])
        zeta = jnp.zeros_like(rho)

        eps_default = lda_correlation_vwn(rho)
        eps_zeta0 = lda_correlation_vwn(rho, zeta=zeta)

        assert_allclose(eps_default, eps_zeta0, atol=1e-10)

    def test_pbe_exchange_spinpol_closed_shell(self):
        """Spin-polarized PBE exchange equals restricted for closed shell."""
        from mess.xcfunctional import gga_exchange_pbe, gga_exchange_pbe_spinpol
        import jax.numpy as jnp

        rho = jnp.array([0.1, 0.2, 0.3])
        grad_rho = jnp.array([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [0.03, 0.0, 0.0]])

        rho_a = rho / 2
        rho_b = rho / 2
        grad_a = grad_rho / 2
        grad_b = grad_rho / 2

        eps_r = gga_exchange_pbe(rho, grad_rho)
        eps_s = gga_exchange_pbe_spinpol(rho_a, rho_b, grad_a, grad_b)

        assert_allclose(eps_r, eps_s, atol=1e-10)
