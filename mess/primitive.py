"""Primitive Gaussian type orbitals"""

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.scipy.special import gammaln

from mess.types import Float3, FloatN, FloatNx3, Int3, asintarray, asfparray


class Primitive(eqx.Module):
    center: Float3 = eqx.field(converter=asfparray, default=(0.0, 0.0, 0.0))
    alpha: float = eqx.field(converter=asfparray, default=1.0)
    lmn: Int3 = eqx.field(converter=asintarray, default=(0, 0, 0))
    norm: Optional[float] = None

    def __post_init__(self):
        if self.norm is None:
            self.norm = normalize(self.lmn, self.alpha)

    def __check_init__(self):
        names = ["center", "alpha", "lmn", "norm"]
        shapes = [(3,), (), (3,), ()]
        dtypes = [jnp.floating, jnp.floating, jnp.integer, jnp.floating]

        for name, shape, dtype in zip(names, shapes, dtypes):
            value = getattr(self, name)
            if value.shape != shape or not jnp.issubdtype(value, dtype):
                raise ValueError(
                    f"Invalid value for {name}.\n"
                    f"Expecting {dtype} array with shape {shape}. "
                    f"Got {value.dtype} with shape {value.shape}"
                )

    @property
    def angular_momentum(self) -> int:
        return np.sum(self.lmn)

    def __call__(self, pos: FloatNx3) -> FloatN:
        return eval_primitive(self, pos)

    def __hash__(self) -> int:
        values = []
        for k, v in vars(self).items():
            if k.startswith("__") or v is None:
                continue

            values.append(v.tobytes())

        return hash(b"".join(values))


@jit
def normalize(lmn: Int3, alpha: float) -> float:
    L = jnp.sum(lmn)
    N = ((1 / 2) / alpha) ** (L + 3 / 2)
    N *= jnp.exp(jnp.sum(gammaln(lmn + 1 / 2)))
    return N**-0.5


def product(a: Primitive, b: Primitive) -> Primitive:
    alpha = a.alpha + b.alpha
    center = (a.alpha * a.center + b.alpha * b.center) / alpha
    lmn = a.lmn + b.lmn
    c = a.norm * b.norm
    Rab = a.center - b.center
    c *= jnp.exp(-a.alpha * b.alpha / alpha * jnp.inner(Rab, Rab))
    return Primitive(center=center, alpha=alpha, lmn=lmn, norm=c)


def eval_primitive(p: Primitive, pos: FloatNx3) -> FloatN:
    pos = jnp.atleast_2d(pos)
    assert pos.ndim == 2 and pos.shape[1] == 3, "pos must have shape [N,3]"
    pos_translated = pos[:, jnp.newaxis] - p.center
    v = p.norm * jnp.exp(-p.alpha * jnp.sum(pos_translated**2, axis=-1))
    v *= jnp.prod(pos_translated**p.lmn, axis=-1)
    return jnp.squeeze(v)
