import os.path as osp

from jax import config
from jax.experimental.compilation_cache import compilation_cache as cc


def pytest_sessionstart(session):
    cache_dir = osp.expanduser("~/.cache/mess")
    print(f"Initializing JAX compilation cache dir: {cache_dir}")
    cc.set_cache_dir(cache_dir)
    config.update("jax_persistent_cache_min_compile_time_secs", 0.1)


def is_mem_limited():
    # Check if we are running on a limited memory host (e.g. github action)
    import psutil

    total_mem_gib = psutil.virtual_memory().total // 1024**3
    return total_mem_gib < 10
