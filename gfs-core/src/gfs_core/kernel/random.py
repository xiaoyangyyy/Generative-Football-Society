from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(root_seed: int, *parts: object) -> int:
    payload = "\x1f".join([
        str(int(root_seed)), *(str(part) for part in parts),
    ])
    digest = hashlib.blake2s(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def named_rng(root_seed: int, *parts: object) -> np.random.Generator:
    return np.random.default_rng(derive_seed(root_seed, *parts))
