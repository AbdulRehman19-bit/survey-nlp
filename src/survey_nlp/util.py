import hashlib, random
import numpy as np
import torch


def sha(s: str, n: int = 16) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def get_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def center(E: np.ndarray, mu=None) -> np.ndarray:
    """Remove the shared direction all review clauses have in common ("I like it" style wording), then re-normalise.
    Without this, every clause is 0.6-0.9 cosine to every theme and themes barely separate."""
    X = E - (E.mean(0, keepdims=True) if mu is None else mu)
    return (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
