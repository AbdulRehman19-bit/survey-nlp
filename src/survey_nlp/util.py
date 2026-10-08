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


def is_spec(cfg) -> bool:
    """mode: spec = sentiment exactly as PROJECT_SPEC.md (model probabilities, evidence_threshold, mixed_policy) and nothing else.
    Anything else (or no mode) keeps the later additions: negation fix, Like/Dislike question rules, learned word lists."""
    try:
        return str(cfg["mode"]).lower() == "spec"
    except (KeyError, TypeError):
        return False


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


_AXIS = {}


def axis_of(cfg):
    """Unit vector pointing from "negative" towards "positive" wording, or None when themes.sentiment_axis is not configured.
    Clustering on raw embeddings groups clauses by HOW they sound ("not good", "really like") as much as by WHAT they are about, so
    a topic splits into a praise cluster and a complaint cluster. Taking this one direction out of every clause (and of the theme
    seeds) lets clauses about the same thing sit together. The anchors are plain English evaluations with no topic in them."""
    try:
        c = cfg["themes"].get("sentiment_axis")
    except (KeyError, TypeError):
        return None
    if not c:
        return None
    key = (tuple(c["positive"]), tuple(c["negative"]), str(cfg["embed"]["model"]))
    if key not in _AXIS:
        from . import embed
        texts = list(c["positive"]) + list(c["negative"])
        V = embed.encode_cached([sha("axis:" + t.lower()) for t in texts], texts, cfg)
        d = V[:len(c["positive"])].mean(0) - V[len(c["positive"]):].mean(0)
        _AXIS[key] = (d / (np.linalg.norm(d) + 1e-12)).astype(np.float32)
    return _AXIS[key]


def center(E: np.ndarray, mu=None, strength: float = 1.0, axis=None) -> np.ndarray:
    """Remove the shared direction all review clauses have in common ("I like it" style wording), then re-normalise.
    Without this, every clause is 0.6-0.9 cosine to every theme and themes barely separate. With `axis` (see axis_of) the
    positive-vs-negative direction is removed too."""
    # strength < 1 removes only part of the shared direction: the survey's most common topic (usually "taste") sits close to
    # the average clause, so removing all of it leaves that topic with no direction and its clauses end up in no theme.
    X = E - strength * (E.mean(0, keepdims=True) if mu is None else mu)
    if axis is not None:
        X = X - (X @ axis)[:, None] * axis[None, :]
    return (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
