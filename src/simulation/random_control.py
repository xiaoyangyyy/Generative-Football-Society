import os
import random
import numpy as np


def set_global_seed(seed=None):
    """
    Set project-wide RNG seeds for reproducibility.
    Priority: explicit arg > GFS_SEED env var.
    """
    if seed is None:
        env_seed = os.getenv("GFS_SEED", "").strip()
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
