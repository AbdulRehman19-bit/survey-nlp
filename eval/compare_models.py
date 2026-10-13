"""Compares sentiment models against the two human coders in data/Comparison.xlsx (500 beverage answers).

    python eval/compare_models.py overall          # whole-answer models (absa.overall) vs the coders' "Sentiment" column
    python eval/compare_models.py absa             # aspect models vs the coders' per-theme +/-/Neutral columns

Gold = Eleni only. The "Hammad" sheet is NOT independent: its Sentiment column equals twitter-roberta-base-sentiment-latest on every answer
(accuracy 1.000), i.e. it is the pipeline's own output. Agreement with it is printed only as a sanity reference."""
import sys
import time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from survey_nlp.sentiment import _cols, _load, predict_probs         # noqa: E402
from survey_nlp.util import get_device                                # noqa: E402

OVERALL = ["cardiffnlp/twitter-roberta-base-sentiment-latest", "j-hartmann/sentiment-roberta-large-english-3-classes"]
ABSA = ["yangheng/deberta-v3-base-absa-v1.1", "yangheng/deberta-v3-large-absa-v1.1"]
THEMES = {"Flavour": "Flavour", "Packaging": "Packaging", "Colour": "Colour", "Energy": "Energy", "Aftertaste": "Aftertaste",
          "Mouth": "Mouth", "Sweetness": "Sweetness", "Drug": "Drug", "Cough syrup": "Cough syrup", "Design": "Design"}
EXTRA_OVERALL = ["cardiffnlp/twitter-roberta-base-sentiment", "finiteautomata/bertweet-base-sentiment-analysis",
                 "lxyuan/distilbert-base-multilingual-cased-sentiments-student", "nlptown/bert-base-multilingual-uncased-sentiment",
                 "cardiffnlp/twitter-xlm-roberta-base-sentiment"]
EXTRA_ABSA = ["nli:MoritzLaurer/deberta-v3-base-zeroshot-v2.0", "nli:facebook/bart-large-mnli", "phrase:yangheng/deberta-v3-base-absa-v1.1"]
LAB = ["neg", "neu", "pos"]
SHORT = {"negative": "neg", "neutral": "neu", "positive": "pos"}


def sentiment_cols(model):
    """Indices of (neg, neu, pos) for any 3-class model; 5-star models return None (handled by stars())."""
    lab = [str(model.config.id2label[i]).lower() for i in range(model.config.num_labels)]
    if model.config.num_labels == 5:
        return None
    if all(x.startswith(("neg", "neu", "pos")) for x in lab):
        return [next(i for i, x in enumerate(lab) if x.startswith(k)) for k in ("neg", "neu", "pos")]
    if all(x.startswith("label_") for x in lab):
        return [0, 1, 2]                                  # cardiffnlp (older): 0 negative, 1 neutral, 2 positive
    raise ValueError(f"unknown labels {lab}")


def stars(P):
    """1-5 star probabilities -> (neg = 1-2 stars, neu = 3, pos = 4-5)."""
    return np.stack([P[:, :2].sum(1), P[:, 2], P[:, 3:].sum(1)], 1)


def nli_probs(name, texts, aspects, dev):
    """Zero-shot: P(entails "The <aspect> is good.") / P(entails "The <aspect> is bad."); neutral = whatever neither claims."""
    tok, model = _load(name, dev)
    lab = [str(model.config.id2label[i]).lower() for i in range(model.config.num_labels)]
    ent = lab.index("entailment")
    out = []
    for adj in ("good", "bad"):
        P = predict_probs(tok, model, texts, [f"The {a.lower()} is {adj}." for a in aspects], 16, 256, dev)
        out.append(P[:, ent])
    pos, neg = out
    return np.stack([neg, 1 - np.maximum(pos, neg), pos], 1)


def load():
    x = pd.ExcelFile(ROOT / "data" / "Comparison.xlsx")
    return x.parse("Eleni"), x.parse("Hammad")


def report(name, gold, ref, P, secs):
    pred = np.array(LAB)[P.argmax(1)]
    print(f"\n### {name}   ({secs:.0f}s)")
    print(f"  vs Eleni (human)  n={len(P)}  acc={accuracy_score(gold, pred):.3f}  macroF1={f1_score(gold, pred, average='macro', labels=LAB):.3f}"
          f"   per class F1 neg/neu/pos = {np.round(f1_score(gold, pred, average=None, labels=LAB), 2).tolist()}")
    print(f"  agreement with the 'Hammad' sheet (pipeline output, reference only): {np.mean(ref == pred):.3f}")
    print("  confusion (rows Eleni, cols model; neg/neu/pos):")
    print(confusion_matrix(gold, pred, labels=LAB))


def overall():
    e, h = load()
    d = e[["Text", "Sentiment"]].merge(h[["Text", "Sentiment"]], on="Text", suffixes=("_e", "_h")).drop_duplicates("Text")
    ga, gb = d.Sentiment_e.str.lower().map(SHORT).to_numpy(), d.Sentiment_h.str.lower().map(SHORT).to_numpy()
    print(f"{len(d)} answers; Eleni neutral share {np.mean(ga == 'neu'):.1%}")
    dev = get_device("auto")
    for name in (EXTRA_OVERALL if EXTRA else OVERALL):
        t = time.time()
        try:
            tok, model = _load(name, dev)
            i = sentiment_cols(model)
            P = predict_probs(tok, model, d.Text.astype(str).tolist(), None, 32, 256, dev)
            P = stars(P) if i is None else P[:, i]
        except Exception as ex:
            print(f"\n### {name}: skipped ({str(ex)[:120]})")
            continue
        report(name, ga, gb, P, time.time() - t)


def absa():
    e, h = load()
    rows = []
    for name, col in THEMES.items():
        ce = {k: f"{col} ({s})" for k, s in (("pos", "+"), ("neg", "-"), ("neu", "Neutral"))}
        if not all(c in e.columns and c in h.columns for c in ce.values()):
            continue

        def lab(df):
            out = pd.Series(None, index=df.index, dtype=object)
            for k, c in ce.items():
                out[pd.to_numeric(df[c], errors="coerce").fillna(0) == 1] = k
            return out
        m = pd.DataFrame({"Text": e.Text, "a": lab(e)}).merge(pd.DataFrame({"Text": h.Text, "b": lab(h)}), on="Text").drop_duplicates("Text")
        m["b"] = m.b.fillna("neu")
        m = m[m.a.notna()]                               # theme coded by Eleni
        m["theme"] = name
        rows.append(m)
    d = pd.concat(rows, ignore_index=True)
    ga, gb = d.a.to_numpy(), d.b.to_numpy()
    print(f"{len(d)} (answer, theme) pairs coded by Eleni; neutral share {np.mean(ga == 'neu'):.1%}\n{d.theme.value_counts().to_dict()}")
    dev = get_device("auto")
    for name in (EXTRA_ABSA if EXTRA else ABSA):
        t = time.time()
        try:
            if name.startswith("nli:"):
                P = nli_probs(name[4:], d.Text.astype(str).tolist(), d.theme.tolist(), dev)
            else:
                phrase = name.startswith("phrase:")          # same model, aspect written as "the flavour" instead of "Flavour"
                name = name.split(":", 1)[-1]
                tok, model = _load(name, dev)
                ineg, ineu, ipos = _cols(model)
                asp = [f"the {a.lower()}" for a in d.theme] if phrase else d.theme.tolist()
                P = predict_probs(tok, model, d.Text.astype(str).tolist(), asp, 32, 256, dev)[:, [ineg, ineu, ipos]]
        except Exception as ex:
            print(f"\n### {name}: skipped ({str(ex)[:120]})")
            continue
        report(name + (" [aspect as 'the <theme>']" if name.startswith("yangheng") and phrase else ""), ga, gb, P, time.time() - t)


if __name__ == "__main__":
    EXTRA = "--extra" in sys.argv                          # the second round of candidates instead of the first
    {"overall": overall, "absa": absa}[sys.argv[1] if len(sys.argv) > 1 else "overall"]()
