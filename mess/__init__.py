"""MESS: Modern Electronic Structure Simulations

The mess simulation environment can be customised by setting the following environment
variables:
  * ``MESS_ENABLE_FP64``: enables float64 precision. Defaults to ``True``.
  * ``MESS_CACHE_DIR``: JIT compilation cache location. Defaults to ``~/.cache/mess``
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mess-jax")
except PackageNotFoundError:  # Package is not installed
    __version__ = "0.0.0.dev0"

from mess.basis import basisset
from mess.hamiltonian import Hamiltonian, UHamiltonian, minimise, uminimise
from mess.structure import molecule, Structure

__all__ = [
    "molecule",
    "Structure",
    "Hamiltonian",
    "UHamiltonian",
    "minimise",
    "uminimise",
    "basisset",
]


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

    enable_fp64 = parse_bool(os.environ.get("MESS_ENABLE_FP64", "True"))
    config.update("jax_enable_x64", enable_fp64)

    cache_dir = str(os.environ.get("MESS_CACHE_DIR", osp.expanduser("~/.cache/mess")))
    config.update("jax_compilation_cache_dir", cache_dir)
    config.update("jax_persistent_cache_min_compile_time_secs", 0.1)
    config.update("jax_persistent_cache_enable_xla_caches", "all")


_setup_env()
