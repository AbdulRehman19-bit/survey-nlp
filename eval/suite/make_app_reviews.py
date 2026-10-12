"""Generates eval/suite/app_reviews_synthetic.xlsx: short app-store style answers to "what do you like / dislike about the app".

A stand-in for a non-food survey. Seeded, so the file is reproducible. Ten topics, each with several ways to name it and to praise or
criticise it; some answers mention two topics, some are generic remarks. The topic of each answer is NOT written to the file.
    python eval/suite/make_app_reviews.py
"""
import random
from pathlib import Path
import pandas as pd

SEED, N = 7, 600
TOPICS = {
    "login": dict(nouns=["login", "sign-in", "account setup", "password reset", "two-factor code"],
                  like=["is quick", "worked first time", "is painless", "feels secure", "never gives me trouble"],
                  dislike=["keeps failing", "logs me out all the time", "never accepts my password", "is confusing", "takes forever"]),
    "notifications": dict(nouns=["notifications", "alerts", "reminders", "push messages"],
                          like=["arrive right on time", "are helpful", "can be customised easily", "are never annoying"],
                          dislike=["come far too often", "arrive late", "cannot be switched off", "are useless", "buzz all night"]),
    "pricing": dict(nouns=["subscription", "price", "monthly fee", "premium plan", "free trial"],
                    like=["is fair", "is great value", "is cheap for what you get", "is clearly explained", "feels worth it"],
                    dislike=["is too expensive", "keeps going up", "is hidden behind a paywall", "renewed without warning", "is not worth it"]),
    "support": dict(nouns=["customer support", "help team", "support chat", "helpdesk"],
                    like=["answered within minutes", "was friendly", "solved my problem", "was really helpful"],
                    dislike=["never replied", "gave copy-paste answers", "took days to answer", "was rude", "was no help at all"]),
    "speed": dict(nouns=["loading time", "speed", "performance", "start-up time"],
                  like=["is fast", "is impressive", "is snappy", "is great even on my old phone"],
                  dislike=["is painfully slow", "gets worse with every update", "drains my battery", "lags constantly", "is sluggish"]),
    "design": dict(nouns=["design", "layout", "interface", "dark mode", "menus"],
                   like=["is clean", "looks beautiful", "is easy to navigate", "is intuitive", "feels modern"],
                   dislike=["is cluttered", "is hard to read", "keeps changing", "is confusing to navigate", "looks dated"]),
    "crashes": dict(nouns=["app", "update", "latest version", "camera feature"],
                    like=["has been rock solid", "never crashes", "runs without a single bug"],
                    dislike=["crashes every time I open it", "freezes all the time", "is full of bugs", "broke after the update", "keeps closing by itself"]),
    "ads": dict(nouns=["ads", "adverts", "pop-ups", "sponsored banners"],
                like=["are not intrusive", "are rare", "are easy to skip"],
                dislike=["are everywhere", "cover the whole screen", "cannot be closed", "ruin the experience", "play with sound"]),
    "sync": dict(nouns=["sync", "cloud backup", "data transfer", "cross-device sync"],
                 like=["works across all my devices", "just works", "keeps everything up to date", "is seamless"],
                 dislike=["lost my data", "fails half the time", "duplicates everything", "never finishes", "does not work between devices"]),
    "search": dict(nouns=["search", "filters", "search bar", "tagging"],
                   like=["finds things instantly", "is very accurate", "is really useful", "saves me so much time"],
                   dislike=["never finds what I need", "returns random results", "is very limited", "is broken", "is slow and useless"]),
}
SHAPES = ["The {n} {p}.", "{N} {p}.", "I think the {n} {p}.", "Honestly the {n} {p}.", "{N} {p}, to be honest.", "My favourite thing: the {n} {p}.",
          "Overall the {n} {p}."]
SHAPES_DISLIKE = ["The {n} {p}.", "{N} {p}.", "Sadly the {n} {p}.", "Honestly the {n} {p}.", "{N} {p} and it is frustrating.",
                  "The main problem: the {n} {p}.", "Really annoyed that the {n} {p}."]
GENERIC_LIKE = ["Does the job.", "Pretty happy with it overall.", "I use it every day.", "Would recommend to a friend.", "Good app.", "Nothing special but it works."]
GENERIC_DISLIKE = ["Not for me.", "Disappointed overall.", "Could be better.", "I stopped using it.", "Expected more.", "Not great."]


def one(rng, kind):
    topics = list(TOPICS)
    r = rng.random()
    if r < 0.12:
        return rng.choice(GENERIC_LIKE if kind == "Like" else GENERIC_DISLIKE)
    parts = []
    for t in rng.sample(topics, 2 if r > 0.68 else 1):
        d = TOPICS[t]
        n, p = rng.choice(d["nouns"]), rng.choice(d[kind.lower()])
        shape = rng.choice(SHAPES if kind == "Like" else SHAPES_DISLIKE)
        parts.append(shape.format(n=n, N=n[0].upper() + n[1:], p=p))
    if len(parts) == 2 and rng.random() < 0.5:                       # one clause joined by "but" / "and"
        a, b = parts[0].rstrip("."), parts[1][0].lower() + parts[1][1:]
        return f"{a}, {rng.choice(['but', 'and'])} {b}"
    return " ".join(parts)


def main():
    rng = random.Random(SEED)
    rows = [(kind, one(rng, kind)) for kind in ("Like", "Dislike") for _ in range(N // 2)]
    out = Path(__file__).with_name("app_reviews_synthetic.xlsx")
    pd.DataFrame(rows, columns=["Source", "Text"]).to_excel(out, index=False)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
