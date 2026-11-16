from contextlib import contextmanager

try:
    import torch
except ImportError:
    torch = None


@contextmanager
def reproducible(seed: int):
    if torch is None:
        raise RuntimeError("Training requires the ML extra: pip install 'gfs-studio[ml]'")
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)
