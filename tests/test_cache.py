import numpy as np
from survey_nlp.cache import EmbCache

def test_roundtrip(tmp_path):
    c = EmbCache(tmp_path / "e.npz")
    c.add(["a", "b"], np.eye(2, dtype=np.float32)); c.save()
    c2 = EmbCache(tmp_path / "e.npz")
    assert c2.missing(["a", "b", "c"]) == ["c"]
    assert np.allclose(c2.get(["b"]), [[0, 1]])
