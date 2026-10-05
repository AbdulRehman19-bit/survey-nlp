import json
from pathlib import Path
import yaml
from .util import sha


class Cfg:
    def __init__(self, raw: dict):
        self.raw = raw

    @classmethod
    def load(cls, path: str) -> "Cfg":
        return cls(yaml.safe_load(Path(path).read_text()))

    def __getitem__(self, k):
        return self.raw[k]

    def section_hash(self, *keys: str) -> str:
        def pick(k):                                    # "section.key" hashes a single setting
            sec, _, sub = k.partition(".")
            return self.raw.get(sec, {}).get(sub) if sub else self.raw.get(sec)
        blob = json.dumps({k: pick(k) for k in keys}, sort_keys=True, default=str)
        return sha(blob, 10)
