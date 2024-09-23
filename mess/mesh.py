"""Discretised sampling of orbitals and charge density."""

from typing import Optional, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from pyscf import dft
from jax import vjp

from mess.basis import Basis
from mess.interop import to_pyscf
from mess.structure import Structure
from mess.types import FloatN, FloatNx3, FloatNxN, MeshAxes


class Mesh(eqx.Module):
    points: FloatNx3
    weights: Optional[FloatN] = None
    axes: Optional[MeshAxes] = None


def uniform_mesh(
    n: Union[int, Tuple] = 50, b: Union[float, Tuple] = 10.0, ndim: int = 3
) -> Mesh:
    if isinstance(n, int):
        n = (n,) * ndim

    if isinstance(b, float):
        b = (b,) * ndim

    if not isinstance(n, (tuple, list)):
        raise ValueError("Expected an integer ")

    if len(n) != ndim:
        raise ValueError("n must be a tuple with {ndim} elements")

    if len(b) != ndim:
        raise ValueError("b must be a tuple with {ndim} elements")

    axes = [jnp.linspace(-bi, bi, ni) for bi, ni in zip(b, n)]
    points = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
    points = points.reshape(-1, ndim)
    return Mesh(points, axes=axes)


def density(basis: Basis, mesh: Mesh, P: Optional[FloatNxN] = None) -> FloatN:
    P = jnp.diag(basis.occupancy) if P is None else P
    orbitals = basis(mesh.points)
    return jnp.einsum("ij,pi,pj->p", P, orbitals, orbitals)


def density_and_grad(
    basis: Basis, mesh: Mesh, P: Optional[FloatNxN] = None
) -> Tuple[FloatN, FloatNx3]:
    def f(points):
        return density(basis, eqx.combine(points, rest), P)

    points, rest = eqx.partition(mesh, lambda x: id(x) == id(mesh.points))
    rho, df = vjp(f, points)
    grad_rho = df(jnp.ones_like(rho))[0].points
    return rho, grad_rho


def molecular_orbitals(
    basis: Basis, mesh: Mesh, C: Optional[FloatNxN] = None
) -> FloatN:
    C = jnp.eye(basis.num_orbitals) if C is None else C
    orbitals = basis(mesh.points) @ C
    return orbitals


def xcmesh_from_pyscf(structure: Structure, level: int = 3) -> Mesh:
    grids = dft.gen_grid.Grids(to_pyscf(structure))
    grids.level = level
    grids.build()
    return Mesh(points=grids.coords, weights=grids.weights)
