r"""Orthonormal transformation.

Evaluates the transformation matrix :math:`X` that satisfies

.. math:: \mathbf{X}^T \mathbf{S} \mathbf{X} = \mathbb{I}

where :math:`\mathbf{S}` is the overlap matrix of the non-orthonormal basis and
:math:`\mathbb{I}` is the identity matrix.

This module implements a few commonly used orthonormalisation transforms.
"""

import jax.numpy as jnp
import jax.numpy.linalg as jnl

from mess.types import FloatNxN


def canonical(S: FloatNxN) -> FloatNxN:
    r"""Canonical orthonormal transformation

    .. math:: \mathbf{X} = \mathbf{U} \mathbf{s}^{-1/2}

    where :math:`\mathbf{U}` and :math:`\mathbf{s}` are the eigenvectors and
    eigenvalues of the overlap matrix :math:`\mathbf{S}`.

    Args:
        S (FloatNxN): overlap matrix for the non-orthonormal basis.

    Returns:
        FloatNxN: canonical orthonormal transformation matrix
    """
    s, U = jnl.eigh(S)
    s = jnp.diag(jnp.power(s, -0.5))
    return U @ s


def symmetric(S: FloatNxN) -> FloatNxN:
    r"""Symmetric orthonormal transformation

    .. math:: \mathbf{X} = \mathbf{U} \mathbf{s}^{-1/2} \mathbf{U}^T

    where :math:`\mathbf{U}` and :math:`\mathbf{s}` are the eigenvectors and
    eigenvalues of the overlap matrix :math:`\mathbf{S}`.

    Args:
        S (FloatNxN): overlap matrix for the non-orthonormal basis.

    Returns:
        FloatNxN: symmetric orthonormal transformation matrix
    """
    s, U = jnl.eigh(S)
    s = jnp.diag(jnp.power(s, -0.5))
    return U @ s @ U.T


def cholesky(S: FloatNxN) -> FloatNxN:
    r"""Cholesky orthonormal transformation

    .. math:: \mathbf{X} = (\mathbf{L}^{-1})^T

    where :math:`\mathbf{L}` is the lower triangular matrix that satisfies the Cholesky
    decomposition of the overlap matrix :math:`\mathbf{S}`.

    Args:
        S (FloatNxN): overlap matrix for the non-orthonormal basis.

    Returns:
        FloatNxN: cholesky orthonormal transformation matrix
    """
    L = jnl.cholesky(S)
    return jnl.inv(L).T
