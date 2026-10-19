import re
import pandas as pd
import spacy
from .util import sha


def run(uniq: pd.DataFrame, cfg) -> pd.DataFrame:
    c = cfg["segment"]
    rx = re.compile(r"[,;]?\s*\b(?:" + "|".join(map(re.escape, c["contrast_words"])) + r")\b\s*", re.I)
    keep_short = c.get("keep_short_next_to_contrast", False)      # "No smell, but delicious": the one-word half is an opinion too
    not_opinion = {w.lower() for w in c.get("short_not_opinion", [])}
    nlp = spacy.blank("en")            # rule-based sentence splitting: no model load, very fast
    nlp.add_pipe("sentencizer")
    rows = []
    texts = uniq.text_clean.tolist()
    for key, doc, raw in zip(uniq.text_key, nlp.pipe(texts, batch_size=1000), texts):
        found = False
        for s in doc.sents:
            parts = rx.split(s.text)
            for part in parts:
                part = part.strip(" ,;.!?-")
                one_word = (keep_short and len(parts) > 1 and part.isalpha() and len(part) >= 3 and part.lower() not in not_opinion)
                if len(part.split()) >= c["min_words"] or one_word:
                    rows.append((key, part))
                    found = True
        if not found:                  # very short answers: keep whole text
            rows.append((key, raw))
    df = pd.DataFrame(rows, columns=["text_key", "clause"])
    df["clause_key"] = [sha(x.lower()) for x in df.clause]
    df["clause_id"] = range(len(df))
    return df
