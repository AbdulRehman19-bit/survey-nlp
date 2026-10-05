from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from .cache import EmbCache
from .util import get_device


def encode_cached(keys, texts, cfg) -> np.ndarray:
    """Unit vectors for `texts` (one per key), embedding only keys missing from the on-disk cache."""
    c = cfg["embed"]
    dev = get_device(cfg["device"])
    cache = EmbCache(Path(c["cache_dir"]) / (c["model"].replace("/", "__") + ".npz"))
    miss = cache.missing(keys)
    if miss:                                    # model only loads if something is new
        text_by_key = dict(zip(keys, texts))
        model = SentenceTransformer(c["model"], device=dev)
        model.max_seq_length = c["max_seq_length"]
        if c["fp16"] and dev == "cuda":
            model.half()
        vecs = model.encode([text_by_key[k] for k in miss], batch_size=c["batch_size"],
                            normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=True).astype(np.float32)   # encode() length-sorts internally
        cache.add(miss, vecs)
        cache.save()
    return cache.get(keys).astype(np.float32)


def run(clauses, cfg) -> np.ndarray:
    return encode_cached(clauses["clause_key"].tolist(), clauses["clause"].tolist(), cfg)
