import pandas as pd
from survey_nlp import suggest


def _df(rows):
    return pd.DataFrame(rows, columns=["asked", "themes", "mixed", "weak_names", "below_bar", "near_duplicates",
                                       "name_coverage_min", "name_coverage_mean", "coherence", "smallest_theme"])


def test_picks_tightest_clean_count():
    df = _df([(8, 8, 1, 0, 0, 0, .3, .6, -.1, 20), (10, 10, 0, 0, 0, 0, .3, .6, -.4, 20),
              (12, 12, 0, 0, 0, 0, .3, .6, -.2, 20), (14, 14, 0, 0, 0, 1, .3, .6, -.1, 20)])
    assert suggest.recommend(df)[0] == 12


def test_fewest_problems_when_none_clean():
    df = _df([(6, 6, 2, 0, 0, 0, .3, .5, -.3, 20), (8, 8, 1, 0, 0, 0, .3, .5, -.5, 20)])
    assert suggest.recommend(df)[0] == 8


def test_empty():
    assert suggest.recommend(_df([]))[0] is None


def test_default_range():
    assert suggest.default_range(553) == (4, 16) and suggest.default_range(100) == (4, 5)


SWEEP = """4	4	0	0	0	0	0.29	0.69	-0.23	109	0.15	0.03	0.33	0.1
5	5	0	0	0	0	0.45	0.74	-0.43	92	0.14	0.03	0.24	0.07
6	6	0	0	0	0	0.45	0.72	-0.5	65	0.11	0.03	0.22	0.11
7	7	0	0	0	0	0.26	0.65	-0.34	47	0.12	0.03	0.24	0.2
8	8	0	0	0	0	0.26	0.63	-0.34	34	0.08	0.02	0.24	0.24
9	9	0	0	0	0	0.32	0.64	-0.39	19	0.06	0.02	0.25	0.29
10	10	0	0	0	0	0.36	0.69	-0.53	27	0.07	0.02	0.23	0.28
11	11	2	0	0	1	0.08	0.5	-0.47	21	0.03	0.02	0.21	0.27
12	12	2	0	0	1	0.08	0.49	-0.47	20	0.03	0.02	0.21	0.31"""


def _sweep(n_clauses=533):
    rows = [[float(x) if "." in x else int(x) for x in r.split("	")] for r in SWEEP.splitlines()]
    df = pd.DataFrame(rows, columns=["asked"] + suggest.COLUMNS[:-1])
    df["smallest_share"] = df.smallest_theme / n_clauses
    return df


def test_real_sweep_is_not_the_smallest_count():
    best, why = suggest.recommend(_sweep())
    assert best == 9 and "Not Specific" in why                      # 4 and 5 have a smallest theme of 17-20% of the file and 14-15% Not Specific


def test_broad_smallest_theme_rejected():
    df = _sweep()
    assert 4 not in suggest.recommend(df)[1].split()[:1] and suggest.recommend(df)[0] != 4
    only_broad = df[df.asked.isin([4, 5])]
    assert suggest.recommend(only_broad)[0] in (4, 5) and "too broad" in suggest.recommend(only_broad)[1]    # no clean count: fewest problems


def test_too_many_unexplained_rejected():
    df = _sweep()
    df.loc[df.asked == 9, "not_specific"] = 0.40
    assert suggest.recommend(df)[0] == 8


def test_weak_name_and_under_bar_count_once():
    cols = ["asked", "themes", "mixed", "weak_names", "below_bar", "near_duplicates", "name_coverage_min", "name_coverage_mean",
            "coherence", "smallest_theme"]
    rows = [(8, 8, 1, 0, 0, 0, .2, .5, -.3, 30), (13, 13, 0, 1, 1, 0, .2, .5, -.1, 30)]       # each has one bad theme (the second is counted by two columns)
    best, why = suggest.recommend(pd.DataFrame(rows, columns=cols))
    assert best == 13 and "fewest (1:" in why
