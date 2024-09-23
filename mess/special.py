"""Special mathematical functions not readily available in JAX."""

from functools import partial
from itertools import combinations_with_replacement

import jax.numpy as jnp
import numpy as np
from jax import lax, vmap
from jax.ops import segment_sum
from jax.scipy.special import betaln, gammainc, gammaln, erfc

from mess.types import FloatN, IntN
from mess.units import LMAX


def factorial_fori(n: IntN, nmax: int = LMAX) -> IntN:
    def body_fun(i, val):
        return val * jnp.where(i <= n, i, 1)

    return lax.fori_loop(1, nmax + 1, body_fun, jnp.ones_like(n))


def factorial_gamma(n: IntN) -> IntN:
    """Appoximate factorial by evaluating the gamma function in log-space.

    This approximation is exact for small integers (n < 10).
    """
    approx = jnp.exp(gammaln(n + 1))
    return jnp.rint(approx)


def factorial_lookup(n: IntN, nmax: int = LMAX) -> IntN:
    N = np.cumprod(np.arange(1, nmax + 1))
    N = np.insert(N, 0, 1)
    N = jnp.array(N, dtype=jnp.uint32)
    return N.at[n.astype(jnp.uint32)].get()


factorial = factorial_gamma


def factorial2_fori(n: IntN, nmax: int = 2 * LMAX) -> IntN:
    def body_fun(i, val):
        return val * jnp.where((i <= n) & (n % 2 == i % 2), i, 1)

    return lax.fori_loop(1, nmax + 1, body_fun, jnp.ones_like(n))


def factorial2_lookup(n: IntN, nmax: int = 2 * LMAX) -> IntN:
    stop = nmax + 1 if nmax % 2 == 0 else nmax + 2
    N = np.arange(1, stop).reshape(-1, 2)
    N = np.cumprod(N, axis=0).reshape(-1)
    N = np.insert(N, 0, 1)
    N = jnp.array(N)
    n = jnp.maximum(n, 0)
    return N.at[n].get()


factorial2 = factorial2_lookup


def binom_beta(x: IntN, y: IntN) -> IntN:
    approx = 1.0 / ((x + 1) * jnp.exp(betaln(x - y + 1, y + 1)))
    return jnp.rint(approx)


def binom_fori(x: IntN, y: IntN, nmax: int = LMAX) -> IntN:
    bang = partial(factorial_fori, nmax=nmax)
    c = x * bang(x - 1) / (bang(y) * bang(x - y))
    return jnp.where(x == y, 1, c)


def binom_lookup(x: IntN, y: IntN, nmax: int = LMAX) -> IntN:
    bang = partial(factorial_lookup, nmax=nmax)
    c = x * bang(x - 1) / (bang(y) * bang(x - y))
    return jnp.where(x == y, 1, c)


binom = binom_lookup


def gammanu_gamma(nu: IntN, t: FloatN, epsilon: float = 1e-10) -> FloatN:
    """Eq 2.11 from THO but simplified using SymPy and converted to jax

        t, u = symbols("t u", real=True, positive=True)
        nu = Symbol("nu", integer=True, nonnegative=True)

        expr = simplify(integrate(u ** (2 * nu) * exp(-t * u**2), (u, 0, 1)))
        f = lambdify((nu, t), expr, modules="scipy")
        ?f

    We evaulate this in log-space to avoid overflow/nan
    """
    t = jnp.maximum(t, epsilon)
    x = nu + 0.5
    gn = jnp.log(0.5) - x * jnp.log(t) + jnp.log(gammainc(x, t)) + gammaln(x)
    return jnp.exp(gn)


def gammanu_series(nu: IntN, t: FloatN, num_terms: int = 128) -> FloatN:
    """Eq 2.11 from THO but simplified as derived in equation 19 of gammanu.ipynb"""
    an = nu + 0.5
    tn = 1 / an
    total = 1 / an

    for _ in range(num_terms):
        an = an + 1
        tn = tn * t / an
        total = total + tn

    return 0.5 * jnp.exp(-t) * total


def gammanu_frac_vmap(nu: IntN, t: FloatN, num_terms: int = 128) -> FloatN:
    def scalar_fn(nu):
        n = jnp.arange(1, num_terms + 1, dtype=t.dtype)
        terms = jnp.where(nu >= n, t**n / jnp.cumprod(n - 0.5), 0.0)
        q = erfc(jnp.sqrt(t)) + jnp.exp(-t) / jnp.sqrt(jnp.pi * t) * jnp.sum(terms)
        lnout = (
            jnp.log(0.5) - (nu + 0.5) * jnp.log(t) + jnp.log(1 - q) + gammaln(nu + 0.5)
        )
        return jnp.exp(lnout)

    out_flt = vmap(scalar_fn)(nu.reshape(-1))
    return out_flt.reshape(nu.shape)


def gammanu_lax_series(nu: IntN, t: FloatN) -> FloatN:
    def cond_fn(vals):
        return jnp.any(vals[0])

    def body_fn(vals):
        enabled, an, tn, total = vals
        an = an + 1
        tn = tn * (t / an)
        total = total + tn
        enabled = enabled & ((tn / total) > jnp.finfo(t.dtype).eps)
        return (enabled, an, tn, total)

    a0 = nu + 0.5
    t0 = 1.0 / a0
    total = 1.0 / t0

    init_vals = (
        jnp.ones(nu.shape, dtype=bool),
        a0,
        t0,
        total,
    )

    _, _, _, total = lax.while_loop(cond_fn, body_fn, init_vals)
    return jnp.exp(jnp.log(0.5) + jnp.log(total) - t)


def gammanu_lax_frac(nu: IntN, t: FloatN) -> FloatN:
    def cond_fn(vals):
        return jnp.any(vals[0])

    def body_fn(vals):
        enabled, term, n, q = vals
        term = term * t / (n - 0.5)
        enabled = nu >= n
        q = jnp.where(enabled, q + term, q)
        n = n + 1.0
        return (enabled, term, n, q)

    enabled = jnp.ones(nu.shape, dtype=bool)
    term = jnp.full(nu.shape, jnp.exp(-t) / jnp.sqrt(jnp.pi * t))
    n = jnp.ones(nu.shape)
    q = jnp.full(nu.shape, erfc(jnp.sqrt(t)))
    init_vals = (enabled, term, n, q)
    _, _, _, q = lax.while_loop(cond_fn, body_fn, init_vals)
    lnout = jnp.log(0.5) - (nu + 0.5) * jnp.log(t) + jnp.log(1 - q) + gammaln(nu + 0.5)
    return jnp.exp(lnout)


def gammanu_select(nu: IntN, t: FloatN, threshold: float = 50.0) -> FloatN:
    """Select between different implementation strategies for evaluation of gammanu

    Args:
        nu (IntN):
        t (FloatN):

    Returns:
        FloatN
    """
    y0 = 1 / (2 * nu + 1)
    y1 = gammanu_series(nu, jnp.minimum(t, threshold))
    y2 = (
        factorial2(2 * nu - 1)
        / 2 ** (nu + 1)
        * jnp.sqrt(jnp.pi / jnp.where(t >= threshold, t, 1.0) ** (2 * nu + 1))
    )

    return jnp.select(
        (t == 0, t < threshold, t >= threshold),
        (y0, y1, y2),
    )


gammanu = gammanu_select


def triu_indices(n: int):
    out = []
    for i in range(n):
        for j in range(i, n):
            out.append((i, j))
    i, j = np.asarray(out).T
    return i, j


def tril_indices(n: int):
    out = []
    for i in range(n):
        for j in range(i + 1):
            out.append((i, j))
    i, j = np.asarray(out).T
    return i, j


def allpairs_indices(n: int):
    pairs_gen = combinations_with_replacement(range(n), r=2)
    i, j = np.asarray(list(pairs_gen)).T
    return i, j


def binom_factor(i: int, j: int, a: float, b: float, lmax: int = LMAX) -> FloatN:
    """Eq. 15 from Augspurger JD, Dykstra CE. General quantum mechanical operators. An
    open-ended approach for one-electron integrals with Gaussian bases. Journal of
    computational chemistry. 1990 Jan;11(1):105-11.
    <https://doi.org/10.1002/jcc.540110113>
    """
    s, t = tril_indices(lmax + 1)
    apow = jnp.maximum(i - (s - t), 0)
    bpow = jnp.maximum(j - t, 0)
    out = binom(i, s - t) * binom(j, t) * a**apow * b**bpow
    mask = ((s - i) <= t) & (t <= j)
    out = jnp.where(mask, out, 0.0)
    return segment_sum(out, s, num_segments=lmax + 1)
