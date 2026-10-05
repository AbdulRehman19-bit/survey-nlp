# Survey NLP: open-ended survey answers to themes and per-theme sentiment

Built from scratch (there is no existing codebase). `Sentiment_Analysis.pptx` is only the design reference for the method; this spec improves on it for speed and rigor.

**Input:** a survey `.xlsx` where one or more columns contain open-ended answers.
**Output:** an `.xlsx` with, for every open-ended column, an overall sentiment column and **one column per discovered theme** coded:

| Code | Meaning |
|---|---|
| 0 | Neutral |
| 1 | Positive |
| 2 | Negative |
| blank | Theme not mentioned in that answer (configurable, see `output.not_discussed_value`) |

There are no separate "+" and "-" columns. The raw probabilities live on a separate detail sheet.

## 0. Kickoff prompt for Claude Code

Put this file and the survey xlsx in an empty repo (`data/survey.xlsx`), start Claude Code, and paste:

> There is no existing code; build everything from scratch. Read PROJECT_SPEC.md fully and build exactly as specified, following the build order in section 12. First run the column auto-detection on data/survey.xlsx and show me which columns it picked before continuing. After each stage run the tests and show me the output. Do not change model names or thresholds without telling me. Ask before adding any dependency not in requirements.txt.

## 1. Design principles

- **Work unit = clause, not response x theme.** Aspect sentiment runs only on (clause, assigned theme) pairs, about 1-2 per clause, instead of every response against every theme.
- **Embed once, cache forever.** Each unique clause is embedded one time and cached to disk by hash.
- **Dedupe before every model call:** texts, clauses, and (clause, theme) pairs.
- **Themes from clause embeddings** (UMAP + HDBSCAN, silhouette-chosen agglomerative as fallback), named with class-based TF-IDF. No noun-chunk rules, no fixed phrase lists.
- **One relevance step:** a single matrix multiply of clause vectors against theme centroids, plus a whole-word keyword fallback.
- **Every stage is checkpointed** and keyed on the config + input file, so changing a threshold re-runs only what depends on it.
- **Thresholds are calibrated on a small hand-labeled sample**, not guessed.

Speed is something to measure, not assume: `timings.json` is written on every run.

## 2. Setup

Python 3.11. GPU optional (CUDA is much faster; CPU works).

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python -m spacy download en_core_web_sm
python scripts/download_models.py                      # one-time; afterwards the pipeline runs offline
pytest -q
```

`requirements.txt`:

```
torch
transformers>=4.40
sentence-transformers>=3.0
sentencepiece
protobuf
spacy>=3.7
scikit-learn>=1.3
umap-learn
scipy
numpy
pandas
pyarrow
openpyxl
pyyaml
tqdm
pytest
```

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "survey-nlp"
version = "0.3.0"
requires-python = ">=3.10"

[project.scripts]
survey-nlp = "survey_nlp.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
```

`scripts/download_models.py`:

```python
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import yaml

cfg = yaml.safe_load(open("config.yaml"))
SentenceTransformer(cfg["embed"]["model"])
for k in ("overall", "absa"):
    AutoTokenizer.from_pretrained(cfg[k]["model"])
    AutoModelForSequenceClassification.from_pretrained(cfg[k]["model"])
print("models cached")
```

## 3. Project layout

```
survey-nlp/
  PROJECT_SPEC.md
  config.yaml
  requirements.txt
  pyproject.toml
  data/survey.xlsx           # gitignored
  cache/                     # embedding cache (gitignored)
  runs/latest/               # stage artifacts + final xlsx (gitignored)
  scripts/download_models.py
  src/survey_nlp/
    __init__.py
    util.py  config.py  cache.py
    ingest.py  segment.py  embed.py  themes.py  assign.py  sentiment.py  export.py
    pipeline.py  cli.py
  eval/
    make_label_sheet.py  evaluate.py  theme_stability.py
  tests/
    test_ingest.py  test_segment.py  test_assign.py  test_sentiment.py  test_cache.py
```

## 4. Data contract

**Input xlsx:** one row per respondent. Any number of open-ended columns (named explicitly in config, or auto-detected). Other columns (ID, demographics, ratings) are never used in NLP; the ones listed in `input.keep_cols` are copied to the output unchanged.

Internally the open-ended columns are melted to long format: one row per (respondent, question).

| Artifact (pickle in `runs/latest/`) | Contents |
|---|---|
| `ingest` | tuple `(long, base)`. `long`: `row_id, respondent_id, question, text, text_clean, valid, text_key`. `base`: one row per respondent with `row_id, respondent_id` + `keep_cols` |
| `segment` | clauses: `clause_id, text_key, clause, clause_key` |
| `embed` | `float32 [n_clauses, 384]`, L2-normalised, aligned to `segment` rows |
| `themes` | dict `scope -> (themes: list[dict], C: float32 [n_themes, 384])`. `scope` is `"ALL"` (pooled) or the question name (per_question) |
| `assign` | `clause_id, theme_idx, sim, via, scope` |
| `overall` | per unique text: `text_key, p_neg, p_neu, p_pos, label, confidence` |
| `absa` | per (clause, theme) pair with `p_neg, p_neu, p_pos`, plus `scope`, `theme`, `clause` |

**Final output** `runs/latest/aspect_sentiment_results.xlsx`:

- `results`: one row per respondent. Columns: `row_id, respondent_id`, the keep columns, then for each question: the original answer text, `<question> | Overall` (0/1/2), and `<question> | <Theme>` (0/1/2/blank) per theme. With only one question the prefix is dropped and columns are just `<Theme>`.
- `detail`: per (respondent, question, theme): `p_pos, p_neg, p_neu, mixed` (1 if both positive and negative evidence fired), `code`.
- `pairs`: clause-level evidence (clause, theme, similarity, probabilities).
- `themes`: names, keywords, sizes. `timings`: seconds per stage.

Also `discovered_themes.json` and `timings.json`.

Example (`results` sheet, two questions, pooled themes):

| respondent_id | Likes | Likes \| Overall | Likes \| Flavour | Likes \| Packaging | Dislikes | Dislikes \| Overall | Dislikes \| Flavour | Dislikes \| Packaging |
|---|---|---|---|---|---|---|---|---|
| 101 | Great taste, sleek bottle | 1 | 1 | 1 | Too sweet | 2 | 2 | |

## 5. config.yaml

```yaml
run_dir: runs/latest
seed: 42
device: auto            # auto | cpu | cuda | mps

input:
  path: data/survey.xlsx
  sheet: 0
  id_col: null                 # respondent ID column; null -> row number
  text_cols: auto              # "auto" or an explicit list, e.g. ["Q5 What do you like?", "Q6 What do you dislike?"]
  keep_cols: []                # extra columns copied to the output untouched (e.g. ["Age", "City"])
  min_chars: 2
  non_answer_regex: "^(n/?a|none|nothing|no comment|nil|-+)$"   # "" to disable
  auto_detect:                 # only used when text_cols: auto
    min_mean_words: 4
    min_unique_ratio: 0.3

segment:
  contrast_words: [but, however, although, though, yet, whereas]
  min_words: 2

embed:
  model: BAAI/bge-small-en-v1.5
  batch_size: 128
  max_seq_length: 128
  fp16: true
  cache_dir: cache

themes:
  scope: pooled                # pooled: one shared theme list for all questions (stabler, comparable)
                               # per_question: separate themes per question (needs a few hundred answers each)
  spacy_model: en_core_web_sm
  min_words: 2
  max_fit_clauses: 20000
  fallback_max_clauses: 4000
  umap_dims: 5
  umap_neighbors: 15
  min_cluster_size: 15
  min_cluster_frac: 0.01
  min_samples: 5
  merge_cosine: 0.85
  min_themes: 2
  max_themes: 10
  min_theme_clauses: 20
  fallback_k_range: [2, 10]
  n_keywords: 8

assign:
  min_sim: 0.50                # STARTING GUESS (clause vs centroid); calibrate (section 9)
  secondary_margin: 0.03
  keyword_min_len: 3

overall:
  model: cardiffnlp/twitter-roberta-base-sentiment-latest
  batch_size: 64
  max_length: 256

absa:
  model: yangheng/deberta-v3-base-absa-v1.1
  batch_size: 64
  max_length: 128
  evidence_threshold: 0.60     # starting guess, recalibrate
  allowance: 0.05              # starting guess, recalibrate
  quantize_cpu: false

output:
  codes: {neutral: 0, positive: 1, negative: 2}
  not_discussed_value: null    # null -> blank cell; set to 0 to treat "not mentioned" as neutral
  mixed_policy: stronger       # when one answer is both positive and negative about a theme:
                               # stronger | positive | negative | neutral
  include_overall: true
```

**Mixed opinions.** One answer such as "great taste but too sweet" can be positive and negative about the same theme, and a single column cannot hold both. `mixed_policy: stronger` picks whichever evidence is stronger. The `detail` sheet marks every mixed case so nothing is hidden.

## 6. Pipeline flow

```
survey.xlsx
  -> ingest      pick open-ended columns, melt to long format, clean, flag empty / non-answers, hash
  -> segment     sentencizer + contrast-word split -> clauses (once)
  -> embed       BGE vectors per unique clause (disk cache)
  -> themes      UMAP + HDBSCAN -> centroid merge -> c-TF-IDF names -> 2-10 themes (per scope)
  -> assign      clause x centroid matmul: top-1 / near-tie top-2 above min_sim, OR whole-word keyword
  -> overall     RoBERTa on unique full texts (independent of themes)
  -> absa        DeBERTa on unique (clause, theme) pairs only
  -> aggregate   max prob across clauses per answer x theme -> code 0 / 1 / 2 (mixed policy applied)
  -> export      xlsx + json + timings
```

## 7. Code

### src/survey_nlp/util.py

```python
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
```

### src/survey_nlp/config.py

```python
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
        blob = json.dumps({k: self.raw.get(k) for k in keys}, sort_keys=True, default=str)
        return sha(blob, 10)
```

### src/survey_nlp/cache.py

```python
from pathlib import Path
import numpy as np


class EmbCache:
    """Disk cache of unit vectors keyed by text hash. One file per embedding model."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.index, self.mat = {}, None
        if self.path.exists():
            z = np.load(self.path, allow_pickle=False)
            self.mat = z["mat"]
            self.index = {k: i for i, k in enumerate(z["keys"].tolist())}

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
        np.savez(self.path, keys=keys, mat=self.mat)
```

### src/survey_nlp/ingest.py

```python
import pandas as pd
from .util import sha


def detect_text_cols(df: pd.DataFrame, c: dict) -> list:
    """Heuristic: string columns with longish, mostly-unique answers are open-ended."""
    d = c.get("auto_detect", {})
    cols = []
    for col in df.columns:
        if col == c.get("id_col"):
            continue
        s = df[col].dropna()
        if s.empty or s.map(lambda x: isinstance(x, str)).mean() < 0.9:
            continue
        mean_words = s.astype(str).str.split().str.len().mean()
        uniq = s.nunique() / len(s)
        if mean_words >= d.get("min_mean_words", 4) and uniq >= d.get("min_unique_ratio", 0.3):
            cols.append(col)
    return cols


def run(cfg):
    c = cfg["input"]
    p = c["path"]
    if p.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(p, sheet_name=c.get("sheet", 0))
    else:
        df = pd.read_csv(p)
    cols = c.get("text_cols", "auto")
    if cols in ("auto", None, []):
        cols = detect_text_cols(df, c)
    missing = [x for x in cols if x not in df.columns]
    if missing:
        raise ValueError(f"text_cols not found in file: {missing}. Available: {list(df.columns)}")
    if not cols:
        raise ValueError("No open-ended columns found. Set input.text_cols explicitly.")
    print("Open-ended columns analysed:", cols)

    ids = df[c["id_col"]].to_numpy() if c.get("id_col") else range(len(df))
    base = pd.DataFrame({"row_id": range(len(df)), "respondent_id": ids})
    for k in c.get("keep_cols", []):
        base[k] = df[k].to_numpy()

    parts = [pd.DataFrame({"row_id": range(len(df)), "respondent_id": ids,
                           "question": q, "text": df[q].to_numpy()}) for q in cols]
    long = pd.concat(parts, ignore_index=True)
    clean = long.text.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    valid = clean.str.len() >= c["min_chars"]
    if c.get("non_answer_regex"):
        valid &= ~clean.str.lower().str.fullmatch(c["non_answer_regex"])
    long["text_clean"] = clean
    long["valid"] = valid
    long["text_key"] = [sha(t.lower()) if v else "" for t, v in zip(clean, valid)]
    return long, base
```

### src/survey_nlp/segment.py

```python
import re
import pandas as pd
import spacy
from .util import sha


def run(uniq: pd.DataFrame, cfg) -> pd.DataFrame:
    c = cfg["segment"]
    rx = re.compile(r"[,;]?\s*\b(?:" + "|".join(map(re.escape, c["contrast_words"])) + r")\b\s*", re.I)
    nlp = spacy.blank("en")            # rule-based sentence splitting: no model load, very fast
    nlp.add_pipe("sentencizer")
    rows = []
    texts = uniq.text_clean.tolist()
    for key, doc, raw in zip(uniq.text_key, nlp.pipe(texts, batch_size=1000), texts):
        found = False
        for s in doc.sents:
            for part in rx.split(s.text):
                part = part.strip(" ,;.!?-")
                if len(part.split()) >= c["min_words"]:
                    rows.append((key, part))
                    found = True
        if not found:                  # very short answers: keep whole text
            rows.append((key, raw))
    df = pd.DataFrame(rows, columns=["text_key", "clause"])
    df["clause_key"] = [sha(x.lower()) for x in df.clause]
    df["clause_id"] = range(len(df))
    return df
```

### src/survey_nlp/embed.py

```python
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from .cache import EmbCache
from .util import get_device


def run(clauses, cfg) -> np.ndarray:
    c = cfg["embed"]
    dev = get_device(cfg["device"])
    cache = EmbCache(Path(c["cache_dir"]) / (c["model"].replace("/", "__") + ".npz"))
    keys = clauses["clause_key"].tolist()
    miss = cache.missing(keys)
    if miss:                                    # model only loads if something is new
        text_by_key = dict(zip(clauses.clause_key, clauses.clause))
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
```

### src/survey_nlp/themes.py

```python
import numpy as np
import spacy
import umap
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def _cluster(X, c, seed):
    """Returns (labels, idx) where idx indexes into X. label -1 = noise."""
    Z = umap.UMAP(n_components=c["umap_dims"], n_neighbors=c["umap_neighbors"],
                  min_dist=0.0, metric="cosine", random_state=seed).fit_transform(X)
    mcs = max(c["min_cluster_size"], int(c["min_cluster_frac"] * len(X)))
    lab = HDBSCAN(min_cluster_size=mcs, min_samples=c["min_samples"]).fit_predict(Z)
    if len(set(lab) - {-1}) >= c["min_themes"]:
        return lab, np.arange(len(X))
    # Fallback: agglomerative, k chosen by silhouette, on a subsample (O(n^2) memory)
    rng = np.random.default_rng(seed)
    sub = np.arange(len(X))
    if len(X) > c["fallback_max_clauses"]:
        sub = rng.choice(len(X), c["fallback_max_clauses"], replace=False)
    Xs = X[sub]
    best, best_s = None, -1.0
    for k in range(c["fallback_k_range"][0], c["fallback_k_range"][1] + 1):
        l = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(Xs)
        s = silhouette_score(Xs, l, metric="cosine", sample_size=min(5000, len(Xs)), random_state=seed)
        if s > best_s:
            best, best_s = l, s
    return best, sub


def _name_themes(clauses, merged, c):
    nlp = spacy.load(c["spacy_model"], disable=["ner", "lemmatizer"])
    docs = [" ".join(clauses["clause"].iloc[m]) for m in merged]
    cv = CountVectorizer(ngram_range=(1, 2), stop_words="english",
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-']+\b")
    tf = cv.fit_transform(docs).toarray().astype(float)
    terms = np.array(cv.get_feature_names_out())
    w = tf / np.maximum(tf.sum(1, keepdims=True), 1)
    A = tf.sum() / len(docs)
    score = w * np.log(1 + A / np.maximum(tf.sum(0), 1))          # class-based TF-IDF
    out, used = [], set()
    for i, m in enumerate(merged):
        order = np.argsort(-score[i])
        kws = terms[order[:c["n_keywords"]]].tolist()
        name = None
        for t in terms[order[:40]]:                                 # POS-guarded: head must be a noun
            last = nlp(t)[-1]
            if last.pos_ in ("NOUN", "PROPN") and t.title() not in used:
                name = t.title()
                break
        name = name or kws[0].title()
        used.add(name)
        out.append({"theme_name": name, "keywords": kws, "frequency": int(len(m))})
    return out


def discover(clauses, E, cfg):
    """clauses must have a 0..n-1 index aligned with E. Returns (themes, C)."""
    c, seed = cfg["themes"], cfg["seed"]
    rng = np.random.default_rng(seed)
    ok = np.where(clauses["clause"].str.split().str.len() >= c["min_words"])[0]
    if len(ok) > c["max_fit_clauses"]:
        ok = rng.choice(ok, c["max_fit_clauses"], replace=False)
    if len(ok) < 2 * c["min_cluster_size"]:
        raise ValueError(f"Only {len(ok)} usable clauses; too few for theme discovery. "
                         "Use themes.scope: pooled or lower themes.min_cluster_size.")
    lab, sub = _cluster(E[ok], c, seed)
    members = {l: ok[sub[lab == l]] for l in sorted(set(lab) - {-1})}
    ls = list(members)
    cents = np.stack([_unit(E[members[l]].mean(0)) for l in ls])
    # merge near-duplicate clusters (connected components over centroid cosine)
    _, comp = connected_components(csr_matrix(cents @ cents.T >= c["merge_cosine"]), directed=False)
    groups = {}
    for l, g in zip(ls, comp):
        groups.setdefault(g, []).extend(members[l].tolist())
    merged = [np.array(v) for v in groups.values() if len(v) >= c["min_theme_clauses"]]
    merged.sort(key=len, reverse=True)
    merged = merged[:c["max_themes"]]
    if len(merged) < c["min_themes"]:
        raise ValueError(f"Only {len(merged)} themes found. Lower themes.min_cluster_size / min_theme_clauses.")
    themes = _name_themes(clauses, merged, c)
    C = np.stack([_unit(E[m].mean(0)) for m in merged]).astype(np.float32)
    return themes, C
```

### src/survey_nlp/assign.py

```python
import re
import numpy as np
import pandas as pd


def run(clauses, E, themes, C, cfg) -> pd.DataFrame:
    """clauses/E may be a subset (a scope); clause_id values are preserved in the output."""
    c = cfg["assign"]
    S = E @ C.T                                         # all clauses x all themes in one matmul
    n = len(S)
    r = np.arange(n)
    order = np.argsort(-S, axis=1)[:, :2] if S.shape[1] > 1 else np.zeros((n, 2), int)
    s1, s2 = S[r, order[:, 0]], S[r, order[:, 1]]
    A = np.zeros(S.shape, bool)
    A[r, order[:, 0]] = s1 >= c["min_sim"]
    sec = (s2 >= c["min_sim"]) & ((s1 - s2) <= c["secondary_margin"]) & (order[:, 1] != order[:, 0])
    A[r[sec], order[sec, 1]] = True
    K = np.zeros_like(A)
    for j, t in enumerate(themes):                      # whole-word keyword match, case-insensitive
        kws = [k for k in t["keywords"] if len(k) >= c["keyword_min_len"]]
        if kws:
            rx = re.compile(r"\b(?:" + "|".join(map(re.escape, kws)) + r")\b", re.I)
            K[:, j] = clauses["clause"].str.contains(rx).to_numpy()
    ii, jj = np.nonzero(A | K)
    via = np.where(A[ii, jj] & K[ii, jj], "both", np.where(A[ii, jj], "semantic", "keyword"))
    return pd.DataFrame({"clause_id": clauses.clause_id.to_numpy()[ii], "theme_idx": jj,
                         "sim": S[ii, jj], "via": via})
```

### src/survey_nlp/sentiment.py

```python
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .util import get_device


def _load(name, dev, fp16=True, quantize=False):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).eval()
    if dev == "cuda" and fp16:
        model = model.half()
    if dev == "cpu" and quantize:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return tok, model.to(dev)


def _cols(model):
    """Column indices for (neg, neu, pos), read from the model config, never hardcoded."""
    m = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
    try:
        return m["negative"], m["neutral"], m["positive"]
    except KeyError:
        raise ValueError(f"Unexpected labels in model config: {model.config.id2label}")


@torch.inference_mode()
def predict_probs(tok, model, a, b=None, batch_size=64, max_length=256, dev="cpu"):
    """Length-sorted batching. `a` = texts; `b` = optional second segment (aspect)."""
    order = np.argsort([len(x) for x in a])
    out = np.zeros((len(a), model.config.num_labels), np.float32)
    for i in range(0, len(a), batch_size):
        ix = order[i:i + batch_size]
        enc = tok([a[j] for j in ix], [b[j] for j in ix] if b is not None else None,
                  padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(dev)
        out[ix] = torch.softmax(model(**enc).logits.float(), -1).cpu().numpy()
    return out


def run_overall(uniq, cfg) -> pd.DataFrame:
    c, dev = cfg["overall"], get_device(cfg["device"])
    tok, model = _load(c["model"], dev)
    ineg, ineu, ipos = _cols(model)
    P = predict_probs(tok, model, uniq.text_clean.tolist(), None, c["batch_size"], c["max_length"], dev)
    P = P[:, [ineg, ineu, ipos]]
    out = pd.DataFrame({"text_key": uniq.text_key.values, "p_neg": P[:, 0], "p_neu": P[:, 1], "p_pos": P[:, 2]})
    out["label"] = np.array(["negative", "neutral", "positive"])[P.argmax(1)]
    out["confidence"] = P.max(1)
    return out


def run_absa(clauses, pairs, themes_by_scope, cfg) -> pd.DataFrame:
    c, dev = cfg["absa"], get_device(cfg["device"])
    df = pairs.merge(clauses[["clause_id", "text_key", "clause_key", "clause"]], on="clause_id")
    df["theme"] = [themes_by_scope[s][j]["theme_name"] for s, j in zip(df.scope, df.theme_idx)]
    key = ["clause_key", "scope", "theme_idx"]
    u = df.drop_duplicates(key)[key + ["clause", "theme"]].reset_index(drop=True)   # score each unique pair once
    tok, model = _load(c["model"], dev, quantize=c["quantize_cpu"])
    ineg, ineu, ipos = _cols(model)
    P = predict_probs(tok, model, u.clause.tolist(), u.theme.tolist(), c["batch_size"], c["max_length"], dev)
    u["p_neg"], u["p_neu"], u["p_pos"] = P[:, ineg], P[:, ineu], P[:, ipos]
    return df.merge(u[key + ["p_neg", "p_neu", "p_pos"]], on=key)


def decide(p_pos, p_neg, threshold, allowance):
    """Binary flags. Both can fire (mixed opinion). Neutral = neither fired."""
    pos = (p_pos >= threshold) & (p_neg <= p_pos + allowance)
    neg = (p_neg >= threshold) & (p_pos <= p_neg + allowance)
    return pos.astype(int), neg.astype(int), (~pos & ~neg).astype(int)


def aggregate(scored, cfg) -> pd.DataFrame:
    """Per (text, scope, theme): max prob across clauses, then one code 0/1/2 using output.codes."""
    c, o = cfg["absa"], cfg["output"]
    codes = o["codes"]
    g = scored.groupby(["text_key", "scope", "theme_idx"])[["p_pos", "p_neg", "p_neu"]].max().reset_index()
    pos, neg, _ = decide(g.p_pos, g.p_neg, c["evidence_threshold"], c["allowance"])
    pos, neg = pos.to_numpy() == 1, neg.to_numpy() == 1
    code = np.full(len(g), codes["neutral"])
    code[pos & ~neg] = codes["positive"]
    code[neg & ~pos] = codes["negative"]
    both = pos & neg
    if o["mixed_policy"] == "stronger":
        code[both] = np.where(g.p_pos.to_numpy()[both] >= g.p_neg.to_numpy()[both], codes["positive"], codes["negative"])
    else:
        code[both] = codes[o["mixed_policy"]]
    g["mixed"] = both.astype(int)
    g["code"] = code
    return g
```

### src/survey_nlp/export.py

```python
import json
from pathlib import Path
import pandas as pd
from . import sentiment


def run(art, cfg, timings):
    rd = Path(cfg["run_dir"])
    o = cfg["output"]
    codes = o["codes"]
    long, base = art["ingest"]
    T = art["themes"]                                   # scope -> (themes, C)
    g = sentiment.aggregate(art["absa"], cfg)
    L = long.merge(art["overall"], on="text_key", how="left")
    L["overall_code"] = L["label"].map(codes)
    questions = list(long.question.unique())
    multi = len(questions) > 1

    out = base.copy()
    detail_rows = []
    for q in questions:
        Lq = L[L.question == q].set_index("row_id")
        pre = f"{q} | " if multi else ""
        out[q] = out.row_id.map(Lq["text"])
        if o["include_overall"]:
            out[f"{pre}Overall"] = out.row_id.map(Lq["overall_code"]).astype("Int64")
        scope = "ALL" if cfg["themes"]["scope"] == "pooled" else q
        themes, _ = T[scope]
        gs = g[g.scope == scope]
        for j, t in enumerate(themes):
            sub = gs[gs.theme_idx == j].set_index("text_key")
            col = Lq["text_key"].map(sub["code"])       # NaN = theme not mentioned
            if o["not_discussed_value"] is not None:
                col = col.where(~(col.isna() & Lq["valid"]), o["not_discussed_value"])
            out[f"{pre}{t['theme_name']}"] = out.row_id.map(col).astype("Int64")
            d = Lq.reset_index()[["row_id", "respondent_id", "text_key"]].merge(sub.reset_index(), on="text_key")
            d["question"], d["theme"] = q, t["theme_name"]
            detail_rows.append(d[["row_id", "respondent_id", "question", "theme", "p_pos", "p_neg", "p_neu", "mixed", "code"]])

    detail = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    pairs = art["absa"][["text_key", "scope", "theme", "clause", "sim", "via", "p_neg", "p_neu", "p_pos"]]
    theme_tbl = pd.concat([pd.DataFrame(t).assign(scope=s) for s, (t, _) in T.items()], ignore_index=True)
    path = rd / "aspect_sentiment_results.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        out.to_excel(xw, sheet_name="results", index=False)
        detail.to_excel(xw, sheet_name="detail", index=False)
        pairs.to_excel(xw, sheet_name="pairs", index=False)
        theme_tbl.to_excel(xw, sheet_name="themes", index=False)
        pd.DataFrame([{"stage": k, "seconds": v} for k, v in timings.items()]).to_excel(xw, sheet_name="timings", index=False)
    (rd / "discovered_themes.json").write_text(json.dumps({s: t for s, (t, _) in T.items()}, indent=2))
    (rd / "timings.json").write_text(json.dumps(timings, indent=2))
    return str(path)
```

Notes for the implementer: `Lq["valid"]` is a boolean Series indexed by `row_id`, aligned with `col`. The `results` sheet must contain only the 0/1/2 coded columns per theme, with no probability columns and no separate positive/negative columns.

### src/survey_nlp/pipeline.py

```python
import os, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
from .config import Cfg
from .util import sha, set_seed
from . import ingest, segment, embed, themes as th, assign as asg, sentiment, export

# (stage, parent stages, config sections that invalidate it)
PLAN = [
    ("ingest",  [],                  ["input"]),
    ("segment", ["ingest"],          ["segment"]),
    ("embed",   ["segment"],         ["embed"]),
    ("themes",  ["embed"],           ["themes", "seed"]),
    ("assign",  ["themes"],          ["assign"]),
    ("overall", ["ingest"],          ["overall"]),
    ("absa",    ["assign"],          ["absa"]),
    ("export",  ["absa", "overall"], ["output"]),
]
ORDER = [p[0] for p in PLAN]


def _file_sig(p):
    s = os.stat(p)
    return f"{p}:{s.st_size}:{int(s.st_mtime)}"


def unique_texts(long):
    return long[long.valid].drop_duplicates("text_key").reset_index(drop=True)


def scopes(cfg, long, clauses):
    """scope name -> boolean mask over clauses. pooled: everything; per_question: that question's texts."""
    if cfg["themes"]["scope"] == "pooled":
        return {"ALL": np.ones(len(clauses), bool)}
    v = long[long.valid]
    return {q: clauses.text_key.isin(set(v[v.question == q].text_key)).to_numpy() for q in v.question.unique()}


def _compute(name, cfg, art, timings):
    long = art["ingest"][0] if "ingest" in art else None
    if name == "ingest":
        return ingest.run(cfg)
    if name == "segment":
        return segment.run(unique_texts(long), cfg)
    if name == "embed":
        return embed.run(art["segment"], cfg)
    if name == "themes":
        out = {}
        for s, m in scopes(cfg, long, art["segment"]).items():
            try:
                out[s] = th.discover(art["segment"][m].reset_index(drop=True), art["embed"][m], cfg)
            except ValueError as e:
                raise ValueError(f"[scope: {s}] {e}")
        return out
    if name == "assign":
        parts = []
        for s, m in scopes(cfg, long, art["segment"]).items():
            themes, C = art["themes"][s]
            a = asg.run(art["segment"][m].reset_index(drop=True), art["embed"][m], themes, C, cfg)
            a["scope"] = s
            parts.append(a)
        return pd.concat(parts, ignore_index=True)
    if name == "overall":
        return sentiment.run_overall(unique_texts(long), cfg)
    if name == "absa":
        return sentiment.run_absa(art["segment"], art["assign"], {s: t for s, (t, _) in art["themes"].items()}, cfg)
    if name == "export":
        return export.run(art, cfg, timings)


def run(config="config.yaml", until="export", force=False):
    cfg = Cfg.load(config)
    set_seed(cfg["seed"])
    rd = Path(cfg["run_dir"])
    rd.mkdir(parents=True, exist_ok=True)
    art, keys, timings = {}, {}, {}
    for name, parents, sections in PLAN:
        seed = [_file_sig(cfg["input"]["path"])] if name == "ingest" else [keys[p] for p in parents]
        keys[name] = sha("|".join(seed + [cfg.section_hash(*sections)]), 12)
        f, kf = rd / f"{name}.pkl", rd / f"{name}.key"
        if name != "export" and not force and f.exists() and kf.exists() and kf.read_text() == keys[name]:
            art[name] = pickle.loads(f.read_bytes())
            timings[name] = 0.0
            print(f"[{name}] cached")
        else:
            t = time.perf_counter()
            art[name] = _compute(name, cfg, art, timings)
            timings[name] = round(time.perf_counter() - t, 2)
            if name != "export":
                f.write_bytes(pickle.dumps(art[name]))
                kf.write_text(keys[name])
            print(f"[{name}] {timings[name]}s")
        if name == until:
            break
    print(f"total {sum(timings.values()):.1f}s")
    return art
```

### src/survey_nlp/cli.py

```python
import argparse
from .pipeline import run, ORDER


def main():
    p = argparse.ArgumentParser("survey-nlp")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", default="config.yaml")
    r.add_argument("--until", default="export", choices=ORDER)
    r.add_argument("--force", action="store_true", help="ignore stage checkpoints (embedding cache is kept)")
    a = p.parse_args()
    run(a.config, a.until, a.force)


if __name__ == "__main__":
    main()
```

Usage: `survey-nlp run`, `survey-nlp run --until ingest` (check which columns were detected), `survey-nlp run --until themes` (inspect themes before sentiment), `survey-nlp run --force`.

## 8. Evaluation (what makes the method defensible)

Hand-label a sample, tune thresholds on it, report the numbers.

### eval/make_label_sheet.py

```python
"""Creates label_sheet.xlsx: ~N assigned pairs (stratified by theme) + near-miss pairs just below the threshold."""
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from survey_nlp.config import Cfg
from survey_nlp.pipeline import scopes

cfg = Cfg.load("config.yaml"); rd = Path(cfg["run_dir"])
L = lambda n: pickle.loads((rd / f"{n}.pkl").read_bytes())
(long, _), clauses, E, T, pairs, scored = L("ingest"), L("segment"), L("embed"), L("themes"), L("assign"), L("absa")
N_ASSIGNED, N_NEAR = 150, 50
n_themes_total = sum(len(t) for t, _ in T.values())
per = max(1, N_ASSIGNED // n_themes_total)

rows = []
for scope, m in scopes(cfg, long, clauses).items():
    themes, C = T[scope]
    sub = clauses[m].reset_index(drop=True)
    Es = E[m]
    sc = scored[scored.scope == scope]
    a = sc.groupby("theme_idx", group_keys=False).apply(lambda d: d.sample(min(per, len(d)), random_state=0))
    rows.append(a.assign(kind="assigned")[["kind", "scope", "clause_id", "clause", "theme_idx", "theme", "sim", "p_pos", "p_neg", "p_neu"]])

    S = Es @ C.T
    lo, hi = cfg["assign"]["min_sim"] - 0.15, cfg["assign"]["min_sim"]
    have = set(zip(pairs[pairs.scope == scope].clause_id, pairs[pairs.scope == scope].theme_idx))
    ii, jj = np.nonzero((S >= lo) & (S < hi))
    cand = [(i, j) for i, j in zip(ii, jj) if (sub.clause_id.iloc[i], j) not in have]
    rng = np.random.default_rng(0)
    k = min(max(1, N_NEAR * len(themes) // n_themes_total), len(cand))
    pick = [cand[x] for x in rng.choice(len(cand), k, replace=False)] if k else []
    rows.append(pd.DataFrame({"kind": "near_miss", "scope": scope,
                              "clause_id": [sub.clause_id.iloc[i] for i, _ in pick],
                              "clause": [sub.clause.iloc[i] for i, _ in pick],
                              "theme_idx": [j for _, j in pick],
                              "theme": [themes[j]["theme_name"] for _, j in pick],
                              "sim": [S[i, j] for i, j in pick]}))
sheet = pd.concat(rows).sample(frac=1, random_state=1).reset_index(drop=True)
sheet["gold_relevant"] = ""      # fill: 1 if the clause is about the theme, else 0
sheet["gold_sentiment"] = ""     # fill (only when relevant): pos / neg / neu
sheet.to_excel(rd / "label_sheet.xlsx", index=False)
print("wrote", rd / "label_sheet.xlsx", len(sheet), "rows")
```

Workflow: run the pipeline, run `python eval/make_label_sheet.py`, fill `gold_relevant` (1/0) and `gold_sentiment` (pos/neg/neu) in the xlsx (about 200 rows, roughly 30-40 minutes), save as `runs/latest/label_sheet_labeled.xlsx`.

### eval/evaluate.py

```python
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
sys.path.insert(0, "src")
from survey_nlp.config import Cfg
from survey_nlp.sentiment import decide

cfg = Cfg.load("config.yaml"); rd = Path(cfg["run_dir"])
df = pd.read_excel(rd / "label_sheet_labeled.xlsx")
df = df[df.gold_relevant.notna() & (df.gold_relevant.astype(str) != "")].copy()
df["gold_relevant"] = df.gold_relevant.astype(int)

print("== Relevance: sweep of assign.min_sim (semantic score only) ==")
best = (0, None)
for t in np.arange(0.35, 0.80, 0.025):
    pred = (df.sim >= t).astype(int)
    p, r, f, _ = precision_recall_fscore_support(df.gold_relevant, pred, average="binary", zero_division=0)
    print(f"min_sim={t:.3f}  P={p:.2f}  R={r:.2f}  F1={f:.2f}")
    if f > best[0]:
        best = (f, round(float(t), 3))
print("best min_sim:", best[1], "F1:", round(best[0], 3))

print("\n== ABSA on relevant, assigned pairs ==")
d = df[(df.kind == "assigned") & (df.gold_relevant == 1) & df.gold_sentiment.notna()].copy()
d["gold_sentiment"] = d.gold_sentiment.astype(str).str.strip().str.lower()
d["pred_argmax"] = np.array(["neg", "neu", "pos"])[d[["p_neg", "p_neu", "p_pos"]].to_numpy().argmax(1)]
print("macro-F1 (argmax):", round(f1_score(d.gold_sentiment, d.pred_argmax, average="macro"), 3))
print(confusion_matrix(d.gold_sentiment, d.pred_argmax, labels=["neg", "neu", "pos"]))

best = (0, None)
for thr in np.arange(0.40, 0.90, 0.05):
    pos, neg, neu = decide(d.p_pos, d.p_neg, thr, cfg["absa"]["allowance"])
    pos, neg = pos.to_numpy().astype(bool), neg.to_numpy().astype(bool)
    pred = np.where(pos & ~neg, "pos", np.where(neg & ~pos, "neg", "neu"))
    f = f1_score(d.gold_sentiment, pred, average="macro")
    print(f"evidence_threshold={thr:.2f}  macro-F1={f:.3f}")
    if f > best[0]:
        best = (f, round(float(thr), 2))
print("best evidence_threshold:", best[1])
```

Copy the best values into `config.yaml`, rerun `survey-nlp run` (cached stages are skipped), and record the final numbers.

### eval/theme_stability.py

```python
"""Re-discovers themes on 80% subsamples; reports how stable the centroids are (matched cosine)."""
import pickle, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, "src")
from survey_nlp.config import Cfg
from survey_nlp.pipeline import scopes
from survey_nlp.themes import discover

cfg = Cfg.load("config.yaml"); rd = Path(cfg["run_dir"])
L = lambda n: pickle.loads((rd / f"{n}.pkl").read_bytes())
(long, _), clauses, E, T = L("ingest"), L("segment"), L("embed"), L("themes")
rng = np.random.default_rng(0)
for scope, m in scopes(cfg, long, clauses).items():
    sub, Es = clauses[m].reset_index(drop=True), E[m]
    _, C = T[scope]
    for i in range(5):
        idx = rng.choice(len(sub), int(0.8 * len(sub)), replace=False)
        cfg.raw["seed"] = i + 1
        t_i, C_i = discover(sub.iloc[idx].reset_index(drop=True), Es[idx], cfg)
        sim = C @ C_i.T
        r, c = linear_sum_assignment(-sim)
        print(f"[{scope}] run {i}: themes={len(t_i)}  mean matched cosine={sim[r, c].mean():.3f}  min={sim[r, c].min():.3f}")
```

Rule of thumb: mean matched cosine near or above 0.9 and a stable theme count means the themes are stable. If not, raise `min_cluster_size`, lower `merge_cosine`, or switch `themes.scope` to `pooled`.

## 9. Calibration checklist

1. Run `survey-nlp run --until ingest` and confirm the right open-ended columns were picked. If not, set `input.text_cols` explicitly.
2. Run `--until themes` and inspect `discovered_themes.json` (names, keywords, sizes). Sanity-check by eye. If names look odd, adjust `themes.min_cluster_size` or `merge_cosine`.
3. Run the full pipeline once with defaults.
4. Label about 200 pairs (section 8) and run `evaluate.py`.
5. Update `assign.min_sim`, `absa.evidence_threshold`, and `absa.allowance` from the sweeps. Rerun.
6. Run `theme_stability.py`.
7. Record `timings.json`, the labeled-sample metrics (ABSA macro-F1, relevance P/R/F1) and the final config in `README.md`.
8. Try `absa.quantize_cpu: true` only on CPU, and keep it only if macro-F1 drops by less than 0.01.

## 10. Tests (tests/)

```python
# tests/test_ingest.py
import pandas as pd
from survey_nlp import ingest

def test_detect_picks_open_ended_only():
    df = pd.DataFrame({
        "id": range(40),
        "gender": ["M", "F"] * 20,
        "likes": [f"I really like the taste number {i} a lot" for i in range(40)],
    })
    cols = ingest.detect_text_cols(df, {"id_col": "id", "auto_detect": {"min_mean_words": 4, "min_unique_ratio": 0.3}})
    assert cols == ["likes"]
```

```python
# tests/test_segment.py
import pandas as pd
from survey_nlp.config import Cfg
from survey_nlp import segment

CFG = Cfg({"segment": {"contrast_words": ["but", "however", "although", "though", "yet", "whereas"], "min_words": 2}})

def _run(texts):
    u = pd.DataFrame({"text_key": [f"k{i}" for i in range(len(texts))], "text_clean": texts})
    return segment.run(u, CFG)

def test_contrast_split():
    out = _run(["The packaging is attractive but the flavour feels artificial."])
    assert out.clause.tolist() == ["The packaging is attractive", "the flavour feels artificial"]

def test_short_text_kept():
    assert _run(["Good"]).clause.tolist() == ["Good"]
```

```python
# tests/test_assign.py
import numpy as np, pandas as pd
from survey_nlp.assign import run

def test_keyword_is_whole_word():
    clauses = pd.DataFrame({"clause_id": [0, 1], "clause": ["I like tea", "great steak"]})
    E = np.array([[1, 0], [1, 0]], np.float32)
    C = np.array([[0, 1]], np.float32)                      # orthogonal: semantic never fires
    themes = [{"theme_name": "Tea", "keywords": ["tea"]}]
    cfg = {"assign": {"min_sim": 0.9, "secondary_margin": 0.03, "keyword_min_len": 3}}
    out = run(clauses, E, themes, C, cfg)
    assert out.clause_id.tolist() == [0]                     # "steak" must not match "tea"
```

```python
# tests/test_sentiment.py
import pandas as pd
from survey_nlp.sentiment import decide, aggregate

CODES = {"neutral": 0, "positive": 1, "negative": 2}

def test_mixed_fires_both():
    pos, neg, neu = decide(pd.Series([0.92]), pd.Series([0.90]), 0.60, 0.05)
    assert (pos.iloc[0], neg.iloc[0], neu.iloc[0]) == (1, 1, 0)

def test_neutral_when_neither():
    pos, neg, neu = decide(pd.Series([0.2]), pd.Series([0.1]), 0.60, 0.05)
    assert (pos.iloc[0], neg.iloc[0], neu.iloc[0]) == (0, 0, 1)

def _scored(pp, pn):
    return pd.DataFrame({"text_key": ["a"], "scope": ["ALL"], "theme_idx": [0],
                         "p_pos": [pp], "p_neg": [pn], "p_neu": [0.0]})

def _cfg(policy):
    return {"absa": {"evidence_threshold": 0.6, "allowance": 0.05},
            "output": {"codes": CODES, "mixed_policy": policy}}

def test_codes():
    assert aggregate(_scored(0.9, 0.05), _cfg("stronger")).code.iloc[0] == 1     # positive
    assert aggregate(_scored(0.05, 0.9), _cfg("stronger")).code.iloc[0] == 2     # negative
    assert aggregate(_scored(0.2, 0.1), _cfg("stronger")).code.iloc[0] == 0      # neutral

def test_mixed_policy():
    assert aggregate(_scored(0.92, 0.90), _cfg("stronger")).code.iloc[0] == 1    # 0.92 > 0.90
    assert aggregate(_scored(0.92, 0.90), _cfg("negative")).code.iloc[0] == 2
    assert aggregate(_scored(0.92, 0.90), _cfg("neutral")).code.iloc[0] == 0
```

```python
# tests/test_cache.py
import numpy as np
from survey_nlp.cache import EmbCache

def test_roundtrip(tmp_path):
    c = EmbCache(tmp_path / "e.npz")
    c.add(["a", "b"], np.eye(2, dtype=np.float32)); c.save()
    c2 = EmbCache(tmp_path / "e.npz")
    assert c2.missing(["a", "b", "c"]) == ["c"]
    assert np.allclose(c2.get(["b"]), [[0, 1]])
```

Tests must not download models or need the real survey file.

## 11. Performance notes

- GPU: `device: auto`; fp16 is applied automatically to the embedding model and the two classifiers.
- CPU: set `torch.set_num_threads` to the physical core count; consider `absa.quantize_cpu` (see calibration step 8).
- The biggest wins are structural: dedupe, one embedding per unique clause, ABSA only on assigned pairs. Do not reintroduce loops over every response x theme.
- A new survey wave re-embeds only new clauses (the cache is keyed by clause hash).
- Optional later: export the two classifiers to ONNX Runtime, only if `timings.json` shows model inference still dominates.

## 12. Build order for Claude Code

1. Scaffold: repo layout, `pyproject.toml`, `requirements.txt`, `config.yaml`, `.gitignore` (`data/ cache/ runs/ .venv/`). `util.py`, `config.py`, `cache.py` + `test_cache.py`.
2. `ingest.py` + `test_ingest.py`. Run `survey-nlp run --until ingest` on the real file and **show me the detected columns and a few sample answers**. Wait for my confirmation, then set `input.text_cols` explicitly.
3. `segment.py` + `test_segment.py`. Print 20 sample clauses.
4. `embed.py`. Run twice and confirm the second run embeds nothing.
5. `themes.py`. Print themes (names, keywords, sizes). Pause and show me.
6. `assign.py` + `test_assign.py`. Print the share of clauses assigned and the per-theme counts.
7. `sentiment.py` + `test_sentiment.py`.
8. `export.py`, `pipeline.py`, `cli.py`. Run end to end, show `timings.json`, and show the first rows of the `results` sheet.
9. `eval/` scripts. Stop and wait for my labeled sheet before tuning thresholds.

Rules for the build:
- Never hardcode model labels; read them from `model.config.id2label`.
- No domain word lists in code (the non-answer regex lives in config).
- Deterministic: seed everywhere via `cfg["seed"]`.
- Every stage logs its duration; do not remove the checkpoint keys.
- If a spec detail conflicts with a library API in the installed versions, fix the code and tell me what changed.

## 13. Acceptance criteria

- `pytest -q` passes without network access.
- A second `survey-nlp run` with no changes finishes in seconds (all stages cached).
- Changing only `absa.evidence_threshold` or `output.mixed_policy` does not re-run segmentation, embedding, or theme discovery.
- Changing `segment.contrast_words` re-runs from `segment` onward and reuses cached embeddings for unchanged clauses.
- The `results` sheet has one row per respondent, one `Overall` column and one column per theme for each open-ended question, containing only 0 / 1 / 2 / blank. There are no separate positive or negative columns and no probability columns.
- `detail` contains the probabilities and a `mixed` flag.
- `evaluate.py` reports ABSA macro-F1 and relevance P/R/F1, and the chosen thresholds are written into `config.yaml`.
- Timings and labeled-sample metrics are recorded in `README.md`.