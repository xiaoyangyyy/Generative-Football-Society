import os
import random
import hashlib
import numpy as np
from src.simulation.runtime import environment_snapshot


def set_global_seed(seed=None):
    """
    Set project-wide RNG seeds for reproducibility.
    Priority: explicit arg > GFS_SEED env var.
    """
    if seed is None:
        env_seed = environment_snapshot().get("GFS_SEED", "").strip()
        if env_seed:
            try:
                seed = int(env_seed)
            except Exception:
                seed = None
    if seed is None:
        return None
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def derive_seed(root_seed: int, *parts: object) -> int:
    """Derive a stable uint32 seed without depending on Python's salted hash()."""
    payload = "\x1f".join([str(int(root_seed)), *(str(part) for part in parts)])
    digest = hashlib.blake2s(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def named_rng(root_seed: int, *parts: object) -> np.random.Generator:
    """Create an isolated deterministic stream for one simulation subsystem."""
    return np.random.default_rng(derive_seed(root_seed, *parts))


def named_py_rng(root_seed: int, *parts: object) -> random.Random:
    """Create an isolated deterministic stdlib stream for one subsystem."""
    return random.Random(derive_seed(root_seed, *parts))
