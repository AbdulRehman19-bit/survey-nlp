import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .util import get_device, is_spec


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


_WORD = re.compile(r"[a-z']{3,}")
_TOK = re.compile(r"[a-z']+|[,;.:!?()]")
_NEGATORS = {"not", "no", "never", "without", "zero", "hardly", "barely", "nothing", "neither", "nor", "cannot",
              "less", "lack", "lacks", "lacking", "fewer"}      # "less refreshing" is the opposite of "refreshing", "less artificial" of "artificial"
_NEG_STOP = {"and", "but", "because", "while", "although", "though", "however", "yet", "or"}   # the negation does not reach past these
_NEG_NOT_NEGATING = {"doubt", "only"}                                                         # "no doubt", "not only"
_NEG_WINDOW = 3                                                                               # "not so / too / as / overly X": up to 3 words back


def _scan(text):
    """[(word, negated)] for every word. A word is negated when a negator ("not", "wasn't", "no", "without", "zero"...)
    stands up to 3 words before it with no comma or "and"/"but" in between: "not so artificial", "wasn't as artificial as"."""
    toks = _TOK.findall(str(text).lower().replace("’", "'").replace("�", "'"))
    out = []
    for i, t in enumerate(toks):
        if t in ",;.:!?()":
            continue
        neg = False
        for j in range(i - 1, max(i - _NEG_WINDOW, 0) - 1, -1):
            p = toks[j]
            if p in _NEG_STOP or p in ",;.:!?()":
                break
            if (p in _NEGATORS or p.endswith("n't")) and toks[j + 1] not in _NEG_NOT_NEGATING:
                neg = True
                break
        out.append((t, neg))
    return out


def _val(w, valence):
    """Valence of a word, also for its noun/adverb forms ("artificiality", "artificially" -> "artificial")."""
    if w in valence:
        return valence[w]
    for suf in ("ity", "ness", "ly"):
        if w.endswith(suf) and w[:-len(suf)] in valence:
            return valence[w[:-len(suf)]]
    return 0


def word_valence(scored, text_questions, cfg, min_support=2, min_conf=0.65, per_question_abs=0.3) -> dict:
    """Words that mean something clearly positive (+1) or negative (-1) wherever they appear ("bad", "unpleasant", "weak").
    Two tests, both needed:
      1. alone ("It is bad.") the sentiment model is clearly on one side. Descriptions that only take a side in context
         ("bit", "much", "too", "cherry") come out neutral or unsure here;
      2. if the word appears in several questions, the clauses containing it are rated that same way in every one of them.
         "sweet" sounds positive alone but is rated both ways across the survey, so it does not count.
    text_questions: text_key -> question."""
    u = scored.groupby(["text_key", "clause_key"]).agg(clause=("clause", "first"), pp=("p_pos", "mean"), pn=("p_neg", "mean")).reset_index()
    u = u.merge(text_questions.drop_duplicates(), on="text_key")
    u["v"] = u.pp - u.pn
    rows = [(w, q, v) for c, q, v in zip(u.clause.str.lower(), u.question, u.v)       # a negated word ("not artificial") says the opposite
            for w in {w for w, neg in _scan(c) if not neg and _WORD.fullmatch(w)}]
    if not rows:
        return {}
    d = pd.DataFrame(rows, columns=["w", "q", "v"])
    n = d.groupby("w").v.count()
    words = n[n >= min_support].index.tolist()
    if not words:
        return {}
    P = run_overall(pd.DataFrame({"text_key": words, "text_clean": [f"It is {w}." for w in words]}), cfg)
    per_q = d.groupby(["w", "q"]).v.mean().unstack()
    out = {}
    for w, pp, pn in zip(P.text_key, P.p_pos, P.p_neg):
        sign = 1 if pp >= min_conf and pp > pn else -1 if pn >= min_conf and pn > pp else 0
        if not sign:
            continue
        seen = per_q.loc[w].dropna()
        if len(seen) > 1 and not ((np.sign(seen) == sign) & (seen.abs() >= per_question_abs)).all():
            continue
        out[w] = sign
    return out


def lexical_flags(texts, valence) -> pd.DataFrame:
    """Per text: plain_pos / plain_neg = uses a clearly positive / negative word; flip_pos = a clearly NEGATIVE word is negated
    ("not so artificial", "no artificial taste"), so the text is really positive; flip_neg = a clearly positive word is negated
    ("not good"). The sentiment models often miss exactly these."""
    t = pd.Series(texts).fillna("").astype(str)
    rows = []
    for x in t:
        f = [False] * 4
        for w, neg in _scan(x):
            s = _val(w, valence)
            if s:
                f[(2 if neg else 0) + (0 if (s == 1) != neg else 1)] = True
        rows.append(f)
    return pd.DataFrame(rows, columns=["plain_pos", "plain_neg", "flip_pos", "flip_neg"], index=t.index)


def evaluative(texts, valence):
    """(says something clearly positive, says something clearly negative) for each text, negation taken into account."""
    f = lexical_flags(texts, valence)
    return f.plain_pos | f.flip_pos, f.plain_neg | f.flip_neg


def phrase_flag(texts, patterns) -> pd.Series:
    """Texts matching any regex of output.complaint_phrases: wording that is a complaint on its own ("too sweet", "not fizzy
    enough", "lacks flavour"), whatever question it answers. The expressions are in the config."""
    t = pd.Series(texts).fillna("").astype(str)
    if not patterns:
        return pd.Series(False, index=t.index)
    rx = re.compile("|".join(f"(?:{p})" for p in patterns), re.I)
    return t.map(lambda x: bool(rx.search(x.replace("’", "'"))))


def lukewarm(texts, patterns, valence, blockers=()) -> pd.Series:
    """Answers that only call the product lukewarm: they match one of output.neutral_phrases ("okay", "nothing special"...) and,
    once those phrases are taken out, say nothing clearly positive or negative ("okay but awful smell" is not lukewarm).
    `patterns` are regexes from the config, so the vocabulary stays out of the code. `blockers` (output.neutral_blockers) are
    regexes for words that turn a remark into a complaint or praise by themselves ("too", "lacks"): if one is left after the
    lukewarm phrases are removed, the answer is not lukewarm."""
    t = pd.Series(texts).fillna("").astype(str)
    if not patterns:
        return pd.Series(False, index=t.index)
    rx = re.compile("|".join(f"(?:{p})" for p in patterns), re.I)
    rest = t.map(lambda x: rx.sub(" ", x))
    hit = t.map(lambda x: bool(rx.search(x)))
    if valence:
        pos, neg = evaluative(rest, valence)
        hit &= ~(pos | neg)
    if blockers:
        brx = re.compile("|".join(f"(?:{b})" for b in blockers), re.I)
        hit &= ~rest.map(lambda x: bool(brx.search(x)))
    return hit


_HEDGE = {"bit", "slightly", "little", "somewhat", "tad"}
_SHORTFALL = {"too", "enough", "less", "more", "not", "but", "although", "though", "than", "expected", "hard", "difficult", "poor"}


def firm_text(texts, min_words=4) -> pd.Series:
    """Texts that can carry a real opinion on their own: not a bare mention ("Its taste") and not a mild remark with no
    shortfall in it ("It tastes a bit sweet"), where a negative score from the model is just as likely a misreading."""
    out = []
    for t in pd.Series(texts).fillna("").astype(str):
        w = [x for x, _ in _scan(t)]
        shortfall = any(x in _SHORTFALL or x.endswith("n't") for x in w)
        out.append(len(w) >= min_words and (shortfall or not any(x in _HEDGE for x in w)))
    return pd.Series(out, index=pd.Series(texts).index)


def lexical_fix(code, f, codes):
    """Correct a model rating when the words contradict it through a negation: "not so artificial" is Positive, "not good" is
    Negative. Only when the text has no plain word pointing the other way, so "not sweet, awful smell" is left alone."""
    code = np.array(code, dtype=float)
    up = (f.flip_pos & ~f.flip_neg & ~f.plain_neg).to_numpy()
    dn = (f.flip_neg & ~f.flip_pos & ~f.plain_pos).to_numpy()
    code[up] = codes["positive"]
    code[dn] = codes["negative"]
    return code, up | dn


def aggregate(scored, cfg, valence=None) -> pd.DataFrame:
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
    g["lex_fix"] = False
    g["firm_neg"] = False
    if "clause" in scored and not is_spec(cfg):  # a clause the model calls Negative with high confidence (used by Like questions)
        fc = o.get("firm_negative_conf", 0.95)
        hit = (firm_text(scored.clause).to_numpy() & (scored.p_neg.to_numpy() >= fc) & (scored.p_neg.to_numpy() > scored.p_pos.to_numpy()))
        k = ["text_key", "scope", "theme_idx"]
        firm = scored[k].assign(firm_neg=hit).groupby(k).firm_neg.any().reset_index()
        g = g.drop(columns="firm_neg").merge(firm, on=k, how="left")
    if valence is not None and not is_spec(cfg):  # does any clause of this text/theme use a clearly positive / negative word?
        f = lexical_flags(scored.clause, valence)
        keys = ["text_key", "scope", "theme_idx"]
        flags = pd.concat([scored[keys].reset_index(drop=True), f.reset_index(drop=True)], axis=1).groupby(keys).any().reset_index()
        g = g.merge(flags, on=keys, how="left")
        g["code"], g["lex_fix"] = lexical_fix(g.code, g, codes)
        g["code"] = g.code.astype(int)
        g["firm_neg"] = g.firm_neg & ~g.flip_pos         # "not so artificial" scored Negative is the negation error, not a complaint
        g["ev_pos"], g["ev_neg"] = g.plain_pos | g.flip_pos, g.plain_neg | g.flip_neg
        g["complaint"] = False
        if o.get("complaint_phrases"):                   # "too sweet" in a Like question is a complaint even when the model is unsure
            hit = phrase_flag(scored.clause, o["complaint_phrases"]).to_numpy()
            keys = ["text_key", "scope", "theme_idx"]
            cf = scored[keys].assign(complaint=hit).groupby(keys).complaint.any().reset_index()
            g = g.drop(columns="complaint").merge(cf, on=keys, how="left")
            g["complaint"] = g.complaint.fillna(False).astype(bool)
    return g
