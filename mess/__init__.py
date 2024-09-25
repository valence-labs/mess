"""MESS: Modern Electronic Structure Simulations

The mess simulation environment can be customised by setting the following environment
variables:
  * ``MESS_ENABLE_FP64``: enables float64 precision. Defaults to ``True``.
  * ``MESS_CACHE_DIR``: JIT compilation cache location. Defaults to ``~/.cache/mess``
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mess")
except PackageNotFoundError:  # Package is not installed
    __version__ = "dev"

from mess.basis import basisset
from mess.hamiltonian import Hamiltonian, minimise
from mess.structure import molecule

__all__ = ["molecule", "Hamiltonian", "minimise", "basisset"]


def parse_bool(value: str) -> bool:
    value = value.lower()

    if value in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif value in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError("Failed to parse {value} as a boolean")


def _setup_env():
    import os
    import os.path as osp

    from jax import config
    from jax.experimental.compilation_cache import compilation_cache as cc

    enable_fp64 = parse_bool(os.environ.get("MESS_ENABLE_FP64", "True"))
    config.update("jax_enable_x64", enable_fp64)

    cache_dir = str(os.environ.get("MESS_CACHE_DIR", osp.expanduser("~/.cache/mess")))
    cc.set_cache_dir(cache_dir)


_setup_env()
