from functools import lru_cache
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from .cache import EmbCache
from .util import get_device


@lru_cache(maxsize=2)
def _model(name, dev, max_seq_length, half):
    """The embedding model, loaded once per process: naming and merging themes embed new words many times in one run, and loading the
    weights from disk each time took longer than encoding the few new texts."""
    model = SentenceTransformer(name, device=dev)
    model.max_seq_length = max_seq_length
    if half:
        model.half()
    return model


def encode_cached(keys, texts, cfg) -> np.ndarray:
    """Unit vectors for `texts` (one per key), embedding only keys missing from the on-disk cache."""
    c = cfg["embed"]
    dev = get_device(cfg["device"])
    cache = EmbCache(Path(c["cache_dir"]) / (c["model"].replace("/", "__") + ".npz"))
    miss = cache.missing(keys)
    if miss:                                    # model only loads if something is new
        text_by_key = dict(zip(keys, texts))
        model = _model(c["model"], dev, c["max_seq_length"], bool(c["fp16"] and dev == "cuda"))
        vecs = model.encode([text_by_key[k] for k in miss], batch_size=c["batch_size"],
                            normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=True).astype(np.float32)   # encode() length-sorts internally
        cache.add(miss, vecs)
        cache.save()
    return cache.get(keys).astype(np.float32)


def run(clauses, cfg) -> np.ndarray:
    return encode_cached(clauses["clause_key"].tolist(), clauses["clause"].tolist(), cfg)
