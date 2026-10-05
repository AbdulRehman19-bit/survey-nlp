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
