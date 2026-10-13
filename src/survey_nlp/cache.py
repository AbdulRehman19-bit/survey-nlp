import os
import zipfile
from pathlib import Path
import numpy as np


class EmbCache:
    """Disk cache of unit vectors keyed by text hash. One file per embedding model."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.index, self.mat = {}, None
        if self.path.exists():
            try:
                z = np.load(self.path, allow_pickle=False)
                mat, keys = z["mat"], z["keys"].tolist()
            except (EOFError, OSError, ValueError, KeyError, zipfile.BadZipFile):   # half-written file (interrupted save, two runs at once)
                self.path.unlink(missing_ok=True)                                  # start empty: the vectors are recomputed
            else:
                self.mat, self.index = mat, {k: i for i, k in enumerate(keys)}

    def missing(self, keys):
        return [k for k in dict.fromkeys(keys) if k not in self.index]

    def add(self, keys, vecs):
        base = 0 if self.mat is None else len(self.mat)
        self.mat = vecs if self.mat is None else np.vstack([self.mat, vecs])
        for i, k in enumerate(keys):
            self.index[k] = base + i

    def get(self, keys):
        return self.mat[[self.index[k] for k in keys]]

    def save(self):
        keys = np.array(sorted(self.index, key=self.index.get), dtype=str)
        tmp = self.path.with_name(f"{self.path.stem}.{os.getpid()}.tmp.npz")      # write aside, then swap in: a reader never sees a half file
        np.savez(tmp, keys=keys, mat=self.mat)
        os.replace(tmp, self.path)
