"""automatic differentiation of atomic orbital integrals"""

from functools import partial
from typing import Callable

import equinox as eqx
import jax.numpy as jnp
from jax import grad, jit, tree, vmap
from jax.ops import segment_sum

from mess.basis import Basis
from mess.integrals import _kinetic_primitives, _nuclear_primitives, _overlap_primitives
from mess.primitive import Primitive
from mess.types import Float3, Float3xNxN


def grad_integrate_primitives(a: Primitive, b: Primitive, operator: Callable) -> Float3:
    def f(center):
        return operator(eqx.combine(center, arest), b)

    acenter, arest = eqx.partition(a, lambda x: id(x) == id(a.center))
    return grad(f)(acenter).center


def grad_overlap_primitives(a: Primitive, b: Primitive) -> Float3:
    return grad_integrate_primitives(a, b, _overlap_primitives)


def grad_kinetic_primitives(a: Primitive, b: Primitive) -> Float3:
    return grad_integrate_primitives(a, b, _kinetic_primitives)


def grad_nuclear_primitives(a: Primitive, b: Primitive, c: Float3) -> Float3:
    def n(lhs, rhs):
        return _nuclear_primitives(lhs, rhs, c)

    return grad_integrate_primitives(a, b, n)


@partial(jit, static_argnums=1)
def grad_integrate_basis(basis: Basis, operator: Callable) -> Float3xNxN:
    def take_primitives(indices):
        p = tree.map(lambda x: jnp.take(x, indices, axis=0), basis.primitives)
        c = jnp.take(basis.coefficients, indices)
        return p, c

    ii, jj = jnp.meshgrid(*[jnp.arange(basis.num_primitives)] * 2, indexing="ij")
    lhs, cl = take_primitives(ii.reshape(-1))
    rhs, cr = take_primitives(jj.reshape(-1))
    out = vmap(operator)(lhs, rhs)

    out = cl * cr * out.T
    out = out.reshape(3, basis.num_primitives, basis.num_primitives)
    out = jnp.rollaxis(out, 1)
    out = segment_sum(out, basis.orbital_index, num_segments=basis.num_orbitals)
    out = jnp.rollaxis(out, -1)
    out = segment_sum(out, basis.orbital_index, num_segments=basis.num_orbitals)
    return jnp.rollaxis(out, -1)


def grad_overlap_basis(basis: Basis) -> Float3xNxN:
    return grad_integrate_basis(basis, grad_overlap_primitives)


def grad_kinetic_basis(basis: Basis) -> Float3xNxN:
    return grad_integrate_basis(basis, grad_kinetic_primitives)


def grad_nuclear_basis(basis: Basis) -> Float3xNxN:
    def n(atomic_number, position):
        def op(pi, pj):
            return atomic_number * grad_nuclear_primitives(pi, pj, position)

        return grad_integrate_basis(basis, op)

    out = vmap(n)(basis.structure.atomic_number, basis.structure.position)
    return jnp.sum(out, axis=0)
