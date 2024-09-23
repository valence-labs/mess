"""(experimental) Gaussian orbital integrals without array padding."""

from functools import partial

import jax.numpy as jnp
from jax import jit

from mess.basis import Basis
from mess.integrals import integrate
from mess.primitive import Primitive, product
from mess.types import FloatNxN


@partial(jit, static_argnums=0)
def overlap_basis_zeropad(basis: Basis) -> FloatNxN:
    def op(a, b):
        return _overlap_primitives_zeropad(a, b, basis.max_L)

    return integrate(basis, op)


@partial(jit, static_argnums=2)
def overlap_context(i: int, j: int, max_L: int):
    from mess.special import binom, factorial2

    def gen():
        for s in range(max_L + 1):
            for t in range(2 * s + 1):
                yield s, t

    s, t = jnp.asarray(list(gen())).T
    mask = (2 * s - i <= t) & (t <= j)
    s = jnp.where(mask, s, 0)
    t = jnp.where(mask, t, 0)
    w = binom(i, 2 * s - t) * binom(j, t) * factorial2(2 * s - 1)
    return s, t, w


def _overlap_primitives_zeropad(a: Primitive, b: Primitive, max_L: int) -> float:
    def overlap_axis(i: int, j: int, a: float, b: float) -> float:
        s, t, w = overlap_context(i, j, max_L)
        out = w * a ** (i - (2 * s - t)) * b ** (j - t) / (2 * p.alpha) ** s
        return jnp.sum(out)

    p = product(a, b)
    pa = p.center - a.center
    pb = p.center - b.center
    out = jnp.power(jnp.pi / p.alpha, 1.5) * p.norm

    for ax in range(3):
        out *= overlap_axis(a.lmn[ax], b.lmn[ax], pa[ax], pb[ax])

    return out
