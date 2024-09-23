"""Core functions for common approximations to the exchange-correlation functional"""

import jax
import numpy as np
import jax.numpy as jnp
import jax.numpy.linalg as jnl
from mess.types import FloatN, FloatNx3


def fzeta(z):
    u = (1 + z) ** (4 / 3) + (1 - z) ** (4 / 3) - 2
    v = 2 * (2 ** (1 / 3) - 1)
    return u / v


def d2fdz20():
    f = jax.grad(jax.grad(fzeta))
    return float(f(0.0))


F2 = d2fdz20()

default_threshold = 1e-15


def lda_exchange(rho: FloatN, threshold: float = default_threshold) -> FloatN:
    mask = rho > threshold
    rho = jnp.where(mask, rho, 0.0)
    Cx = (3 / 4) * (3 / np.pi) ** (1 / 3)
    eps_x = -Cx * rho ** (1 / 3)
    eps_x = jnp.where(mask, eps_x, 0.0)
    return eps_x


def lda_correlation_vwn(
    rho: FloatN, threshold: float = default_threshold, use_rpa: bool = True
) -> FloatN:
    A, x0, b, c = vwn_coefs(use_rpa)

    # avoid divide by zero when rho = 0 by replacing with 1.0
    mask = jnp.abs(rho) > threshold
    rho = jnp.where(mask, rho, 1.0)
    rs = jnp.power(3 / (4 * jnp.pi * rho), 1 / 3).reshape(-1, 1)
    x = jnp.sqrt(rs).reshape(-1, 1)
    X = rs + b * x + c
    X0 = x0**2 + b * x0 + c
    Q = np.sqrt(4 * c - b**2)

    u = jnp.log(x**2 / X) + 2 * b / Q * jnp.arctan(Q / (2 * x + b))
    v = jnp.log((x - x0) ** 2 / X) + 2 * (b + 2 * x0) / Q * jnp.arctan(Q / (2 * x + b))
    ec = A * (u - b * x0 / X0 * v)
    e0, e1, alpha = ec.T
    beta = F2 * (e1 - e0) / alpha - 1
    z = jnp.zeros_like(rho)  # restricted ks, should be rho_up - rho_down
    eps_c = e0 + alpha * fzeta(z) / F2 * (1 + beta * z**4)
    eps_c = jnp.where(mask, eps_c, 0.0)
    return eps_c


def vwn_coefs(use_rpa: bool = True):
    # paramagnetic (eps_0) / ferromagnetic (eps_1) / spin stiffness (alpha)
    A = np.array([0.0310907, 0.5 * 0.0310907, -1 / (6 * np.pi**2)])

    if use_rpa:
        x0 = np.array([-0.409286, -0.743294, -0.228344])
        b = np.array([13.0720, 20.1231, 1.06835])
        c = np.array([42.7198, 101.578, 11.4813])
    else:
        # https://math.nist.gov/DFTdata/atomdata/node5.html
        x0 = np.array([-0.10498, -0.32500, -4.75840e-3])
        b = np.array([3.72744, 7.06042, 1.13107])
        c = np.array([12.9352, 18.0578, 13.0045])
    return A, x0, b, c


def lda_correlation_pw(rho: FloatN, threshold: float = default_threshold) -> FloatN:
    p = np.ones(3)
    A = np.array([0.031091, 0.015545, 0.016887])
    a1 = np.array([0.21370, 0.20548, 0.11125])
    b1 = np.array([7.5957, 14.1189, 10.357])
    b2 = np.array([3.5876, 6.1977, 3.6231])
    b3 = np.array([1.6382, 3.3662, 0.88026])
    b4 = np.array([0.49294, 0.62517, 0.49671])

    # avoid divide by zero when rho = 0 by replacing with 1.0
    mask = jnp.abs(rho) > threshold
    rho = jnp.where(mask, rho, 1.0)
    rs = jnp.power(3 / (4 * jnp.pi * rho), 1 / 3).reshape(-1, 1)
    v = 2 * A * (b1 * jnp.sqrt(rs) + b2 * rs + b3 * rs ** (3 / 2) + b4 * rs ** (p + 1))
    G = -2 * A * (1 + a1 * rs) * jnp.log(1 + 1 / v)
    e0, e1, alpha = G.T
    beta = F2 * (e1 - e0) / alpha - 1
    z = jnp.zeros_like(rho)  # restricted ks, should be rho_up - rho_down
    eps_c = e0 + alpha * fzeta(z) / F2 * (1 + beta * z**4)
    eps_c = jnp.where(mask, eps_c, 0.0)
    return eps_c


def gga_exchange_b88(
    rho: FloatN, grad_rho: FloatNx3, threshold: float = default_threshold
) -> FloatN:
    beta = jnp.asarray(0.0042 * 2 ** (1 / 3))
    # avoid divide by zero when rho = 0 by replacing with 1.0
    mask = jnp.abs(rho) > threshold
    rho = jnp.where(mask, rho, 1.0)
    x = jnl.norm(grad_rho, axis=1) / rho ** (4 / 3)
    d = 1 + 6 * beta * x * jnp.arcsinh(2 ** (1 / 3) * x)
    eps_x = lda_exchange(rho) - beta * rho ** (1 / 3) * x**2 / d
    eps_x = jnp.where(mask, eps_x, 0.0)
    return eps_x


def gga_exchange_pbe(
    rho: FloatN, grad_rho: FloatNx3, threshold: float = default_threshold
) -> FloatN:
    beta = np.asarray(0.066725)  # Eq 4
    mu = beta * np.pi**2 / 3  # Eq 12
    kappa = np.asarray(0.8040)  # Eq 14

    # avoid divide by zero when rho = 0 by replacing with 1.0
    mask = jnp.abs(rho) > threshold
    rho = jnp.where(mask, rho, 1.0)
    kf = (3 * np.pi**2 * rho) ** (1 / 3)
    s = jnl.norm(grad_rho, axis=1) / (2 * kf * rho)
    F = 1 + kappa - kappa / (1 + mu * s**2 / kappa)
    F = jnp.where(mask, F, 0.0)
    return lda_exchange(rho) * F


def gga_correlation_pbe(
    rho: FloatN, grad_rho: FloatNx3, threshold: float = default_threshold
) -> FloatN:
    beta = np.asarray(0.066725)
    gamma = (1 - np.log(2.0)) / np.pi**2
    z = jnp.zeros_like(rho)  # restricted ks, should be (rho_up - rho_down) / rho
    phi = 0.5 * (jnp.power(1 + z, 2 / 3) + jnp.power(1 - z, 2 / 3))
    ec_pw = lda_correlation_pw(rho, threshold)
    # avoid divide by zero when rho = 0 by replacing with 1.0
    mask = jnp.abs(rho) > threshold
    rho = jnp.where(mask, rho, 1.0)
    mask_ecpw = jnp.where(mask, ec_pw, 1.0)
    A = beta / gamma * (jnp.exp(-mask_ecpw / (gamma * phi**3)) - 1) ** -1  # Eq 8
    kf = (3 * np.pi**2 * rho) ** (1 / 3)
    ks = jnp.sqrt(4 * kf / np.pi)
    t = jnl.norm(grad_rho, axis=1) / (2 * phi * ks * rho)
    u = 1 + beta / gamma * t**2 * (1 + A * t**2) / (1 + A * t**2 + A**2 * t**4)
    H = gamma * phi**3 * jnp.log(u)  # Eq 7
    H = jnp.where(mask, H, 0.0)
    return ec_pw + H


def gga_correlation_lyp(
    rho: FloatN, grad_rho: FloatNx3, threshold: float = default_threshold
) -> FloatN:
    a = np.asarray(0.04918)
    b = np.asarray(0.132)
    c = np.asarray(0.2533)
    d = np.asarray(0.349)
    CF = 0.3 * (3 * np.pi**2) ** (2 / 3)

    # avoid divide by zero when rho = 0 by replacing with 1.0
    mask = jnp.abs(rho) > threshold
    rho = jnp.where(mask, rho, 1.0)
    v = 1 + d * rho ** (-1 / 3)
    omega = jnp.exp(-c * rho ** (-1 / 3)) / v * rho ** (-11 / 3)
    delta = c * rho ** (-1 / 3) + d * rho ** (-1 / 3) / v
    g = (1 / 24 + 7 * delta / 72) * rho * jnl.norm(grad_rho, axis=1) ** 2

    eps_c = -a / v - a * b * omega * (CF * rho ** (11 / 3) - g)
    eps_c = jnp.where(mask, eps_c, 0.0)
    return eps_c
