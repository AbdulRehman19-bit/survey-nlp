import re
from collections import Counter

_WORD = re.compile(r"[A-Za-z][A-Za-z]*(?:['’][A-Za-z]+)?")

# Apostrophe left out when typing: "dont like it" must still read as a negation. Not a spelling choice, so always repaired.
_CONTRACTIONS = {"dont": "don't", "doesnt": "doesn't", "didnt": "didn't", "isnt": "isn't", "wasnt": "wasn't", "arent": "aren't",
                 "werent": "weren't", "havent": "haven't", "hasnt": "hasn't", "hadnt": "hadn't", "wouldnt": "wouldn't",
                 "couldnt": "couldn't", "shouldnt": "shouldn't", "cant": "can't", "thats": "that's", "whats": "what's"}


def _match_case(src: str, new: str) -> str:
    if src.isupper() and len(src) > 1:
        return new.upper()
    if src[0].isupper():
        return new[0].upper() + new[1:]
    return new


def _variants(w: str):
    """The same word spelt the other way round (British / American): colour - color, flavours - flavors, realise - realize."""
    yield w.replace("our", "or")
    yield re.sub(r"is(?=e|ed|es|ing|ation)", "iz", w)
    yield re.sub(r"(?<=[aeiou])ll(?=ing|ed)", "l", w)
    yield w.replace("tre", "ter")


def _proper_nouns(texts) -> set:
    """Words written with a capital letter in the middle of a sentence: names and brands, never typos."""
    out = set()
    for t in texts:
        t = str(t)
        for m in _WORD.finditer(t):
            if m.group(0)[0].isupper() and m.start() > 0 and t[:m.start()].rstrip()[-1:] not in ".!?:;\n":
                out.add(m.group(0).lower())
    return out


def find_corrections(texts, c: dict, protect=()) -> dict:
    """{wrong word (lower case): repaired word} learned from the answers themselves.

    A word is changed only when ALL of this holds, so brand names, slang and spellings people meant are left alone:
      - the English dictionary does not know it, and neither its British / American twin;
      - it is rare in this survey (fewer than `spell_min_count` uses), or another spelling of it is used at least 5 times as often:
        a word many people wrote, with nothing to replace it, is not a typo;
      - it has at least `spell_min_len` letters, is not a name written with a capital letter and is not in `protect`
        (the words of your own theme names and keywords);
      - a dictionary word that starts with the same letter is 1 edit away (2 for words of 7+ letters);
      - that word is one this survey already uses, or a really common English word (`spell_min_freq`);
      - when several fit, the one used most in the survey wins, then the most common in English."""
    from spellchecker import SpellChecker
    counts = Counter(w.lower() for t in texts for w in _WORD.findall(str(t)))
    short, long_ = SpellChecker(distance=1), SpellChecker(distance=2)
    protect = {p.lower() for p in protect} | _proper_nouns(texts)
    min_count, min_len, min_freq = c.get("spell_min_count", 3), c.get("spell_min_len", 5), c.get("spell_min_freq", 1e-5)
    known = lambda x: x in short or any(v in short for v in _variants(x))      # a real word, in either spelling
    out = {w: v for w, v in _CONTRACTIONS.items() if w in counts}
    for w in short.unknown([w for w in counts if "'" not in w and "’" not in w and w not in out]):
        if len(w) < min_len or w in protect:
            continue
        if any(v != w and v in short for v in _variants(w)) or re.search(r"is(?:e|ed|es|ing|ation|ations)$", w):
            continue                                        # a British / American twin, or an "-ise" verb: a spelling choice, not a typo
        sp = long_ if len(w) >= 7 else short
        near = sp.edit_distance_2(w) if sp is long_ else sp.edit_distance_1(w)
        used = {x for x in near if counts.get(x, 0) >= min_count and known(x)}               # a word this survey writes often: the likeliest meaning
        cands = {x for x in (set(sp.candidates(w) or ()) | used) if x != w and x.isalpha() and x[0] == w[0]
                 and (counts.get(x, 0) > 0 or sp.word_usage_frequency(x) >= min_freq)}
        if counts[w] >= min_count:                      # written by several people: a typo only if a longer dictionary word is far more used
            cands = {x for x in cands if known(x) and len(w) >= 6 and len(x) >= len(w) and counts.get(x, 0) >= 5 * counts[w]}
        if cands:
            out[w] = max(cands, key=lambda x: (counts.get(x, 0), sp.word_usage_frequency(x)))
    return out


def apply_corrections(text: str, fixes: dict) -> str:
    return _WORD.sub(lambda m: _match_case(m.group(0), fixes[m.group(0).lower()]) if m.group(0).lower() in fixes else m.group(0), text)
