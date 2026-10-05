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
