# Survey NLP: complete project overview

Written for: the project owner and anyone new who needs to understand, run or extend the project. No programming background is needed for sections 1 to 7.
It describes the project as it is in the working tree on 2026-10-12. `PROJECT_SPEC.md` is the original design brief, `README.md` holds
calibration notes and run notes, and `CHANGELOG.md` has the earlier change history. This file is the single place that explains
how everything fits together. All examples marked "real" come from the survey files in this project.

Contents: 1 What it does · 2 Words used here · 3 Input and output · 4 Repository layout · 5 Setup and running · 6 The pipeline step by step ·
7 One answer from start to finish · 8 The quality report · 9 Caching · 10 The web app · 11 Settings · 12 Tests and evaluation ·
13 What was tried and what is on or off · 14 Known limitations · 15 What comes next

---

## 1. What the project does

You give it a survey file with **open-ended answers** (free text, for example "What did you like about this product?"). It:

1. cleans the answers and cuts them into short pieces called **clauses**;
2. finds the **themes** the answers talk about (or uses the ones you name);
3. decides which clauses belong to which theme;
4. rates the **sentiment** of each answer overall and **per theme** (Positive / Negative / Neutral);
5. writes an Excel workbook with one coloured sheet per question, plus detail sheets.

Real example (beverage survey, a "what did you like" question). The answer

> Smell bad but taste ok

becomes: **Overall: Negative · Flavour: Positive · Smell: Negative**. One answer, two themes, two different ratings. That is the whole point: a single score per
answer would hide that the person liked the taste and disliked the smell.

Everything runs on your own machine (no external service). The language models are downloaded once and used offline afterwards.
Language: English only (the models are English).

---

## 2. Words used in this file

| Word | Plain meaning |
|---|---|
| **answer** | what one person wrote for one question |
| **clause** | a short piece of an answer that makes one point. "Strong chocolate flavour but not overpowering" has two clauses |
| **theme** | a topic that many answers talk about, for example Packaging, Price, Crunch. Also called an aspect |
| **embedding** | the meaning of a text turned into a list of numbers (384 of them here), so that texts with similar meaning get similar numbers |
| **similarity (cosine)** | how alike two embeddings are, from about 0 (unrelated) to 1 (same meaning). "I liked the mix of white and chocolate pieces" is 0.72 similar to the theme White Chocolate |
| **centre of a theme** | the average embedding of the clauses in the theme: the theme's "direction" in meaning space |
| **sentiment** | Positive, Negative or Neutral |
| **General comment** | a built-in theme for remarks that name no topic: "I like it", "It is good" |
| **required theme** | a theme you insist on (you name it, optionally with words that signal it) |
| **name coverage** | the share of a theme's clauses that actually contain the theme's name. A theme called "Packaging" where 60% of the clauses say "packaging" has coverage 0.6 |
| **mixed theme** | a theme whose name fits too few of its clauses: it probably glues together several topics |
| **checkpoint** | the saved result of one stage, reused next time if nothing changed |

---

## 3. Input and output

**Input:** an `.xlsx`, `.xls` or `.csv` file. Two layouts are detected automatically:

| Layout | Meaning | Real example |
|---|---|---|
| wide | one row per respondent, one column per open-ended question | the Coco Pops survey: 300 respondents, two questions ("What did you LIKE…", "What did you NOT LIKE…") |
| long | one comment per row; a column says which question it answers and another holds the text | the beverage file: columns `Source` (Like / Dislike) and `Text`, 500 rows |

**Output:** `<run_dir>/aspect_sentiment_results.xlsx` with these sheets:

| Sheet | Content |
|---|---|
| one per question | the answer, Overall sentiment, one column per theme (green = Positive, red = Negative, yellow = Neutral, empty = theme not mentioned) |
| Combined | every question's answers one under the other (when there is more than one question) |
| results, results_codes | side-by-side words and 0/1/2 codes (wide layout only) |
| detail | the model's probabilities for every (answer, theme) and the final rating |
| pairs | every (clause, theme) pair that was scored, with how it was matched |
| themes | the themes: size, main words, example clauses, name coverage and flags |
| quality | a health check of the run: see section 8 |
| spelling | typos that were repaired before the models read the text |
| timings | seconds per stage |

Codes: 0 = Neutral, 1 = Positive, 2 = Negative, blank = theme not mentioned. The words shown in the sheets are configurable (`output.labels`).

Other files in the run folder: `discovered_themes.json`, `timings.json`, `spelling_corrections.json`, `quality.json`, `clusters.json` (saved clusters, see 6.6),
`theme_edits.json` (your theme choices from the command line), and one `.pkl` plus `.key` per stage (checkpoints).

---

## 4. Repository layout

```
app.py                      Streamlit web app
config_default.yaml         generic settings (works on any new survey)
config.yaml                 settings tuned for the Coco Pops survey (older: see section 14)
config_beverage.yaml        settings with a fixed codebook for a beverage survey
config_open_ends.yaml       settings for a long-format open-ended file
PROJECT_SPEC.md             original design brief
README.md, CHANGELOG.md     calibration notes, earlier change history
PROJECT_OVERVIEW.md         this file
src/survey_nlp/
    cli.py                  command line: survey-nlp run ...
    pipeline.py             stage plan, checkpoints and cache keys
    config.py               config loading and per-section hashing
    ingest.py               reading files, layout detection, non-answers, spelling step
    spelling.py             typo repair learned from the answers
    segment.py              splitting answers into clauses
    embed.py, cache.py      embeddings and their disk cache
    themes.py               theme discovery, naming, keywords, curation, required themes
    assign.py               clause to theme assignment
    sentiment.py            overall sentiment, per-theme sentiment, rules on top
    export.py               the Excel workbook
    quality.py              the quality report
    util.py                 hashing, seeds, device, centring, sentiment axis, quiet logging
tests/                      96 tests (pytest)
eval/                       evaluation: run_suite.py, suite/ (datasets), quality_eval.py, calibration scripts
scripts/download_models.py  one-time model download
data/                       survey files (not committed)
runs/                       results, one folder per run (can be deleted)
cache/                      embedding cache (one .npz per embedding model)
```

---

## 5. Setup and running

```bash
python -m venv .venv
.venv\Scripts\activate                     # Windows
pip install -r requirements.txt
pip install -e .
python -m spacy download en_core_web_sm
python scripts/download_models.py          # one time
pytest -q
```

**Command line**

```bash
survey-nlp run --config config_default.yaml --input data/your_file.xlsx
```

| Option | Meaning |
|---|---|
| `--until <stage>` | stop after a stage (`themes` to inspect the themes) |
| `--force` | ignore stage checkpoints (the embedding cache is kept) |
| `--rediscover` | find the themes again instead of reusing the saved clusters |
| `--n-themes N` | how many themes to keep (default 10) |
| `--themes "Packaging,Price"` | required themes, comma separated, skips the prompts |
| `--prompt` / `--no-prompt` | always ask / never ask about themes |
| `--run-dir` | where to write results (default `runs/<file name>` with `--input`) |

Interactive mode asks for required themes and the count, shows the discovered themes, and accepts commands until you press Enter:
`n 8`, `add Price: cost, cheap`, `remove Cough, Cherry`, `merge Easy + Bottle = Opening`, `rename Energy = Refreshing`, `reset`.
Choices are saved in `<run_dir>/theme_edits.json` and offered again next time.

**Web app:** `streamlit run app.py`. See section 10.

**Cleaning stored results:** delete `runs\<run folder>` (stage results and saved clusters) and/or `cache\BAAI__bge-small-en-v1.5.npz` (embeddings; rarely needed).

---

## 6. The pipeline step by step

The work is done in stages. Each stage is saved to disk and reused when nothing it depends on has changed (section 9).

| # | Stage | Module | What it does in one line |
|---|---|---|---|
| 1 | ingest | ingest.py | reads the file, cleans the text, repairs typos, removes non-answers |
| 2 | segment | segment.py | cuts each answer into clauses |
| 3 | embed | embed.py | turns each clause into numbers that capture its meaning |
| 4 | themes | themes.py | finds the themes, names them, picks keywords |
| 5 | assign | assign.py | decides which clause belongs to which theme |
| 6 | overall | sentiment.py | rates each whole answer |
| 7 | absa | sentiment.py | rates each (clause, theme) pair |
| 8 | export | export.py | builds the workbook and the quality report |

### 6.1 Ingest ([ingest.py](src/survey_nlp/ingest.py), [spelling.py](src/survey_nlp/spelling.py))

**Goal:** turn any survey file into one clean table: one row per (person, question) answer.

**Reading the file.** For Excel files the header row is guessed: exports often start with a few rows of metadata (survey id, section names), so the program looks at the
first 15 rows and picks the earliest row with the most distinct text cells. Some exports put a row of question ids right under the header (a single long token with digits,
like `q-id-6abb…-value`); that row is removed.

**Finding the answer columns.** A column counts as open-ended when at least 90% of its cells are text, at least 20% of rows are filled, it is not JSON-like
(`[` or `{` at the start), and its answers are longer than 4 words on average and mostly different from each other (at least 30% unique).
This skips ID columns, yes/no columns and nearly empty "other, please specify" columns.

**Wide or long?** If there is exactly one text column plus one short, repeating, label-like column (2 to 12 different values such as Like / Dislike), the file is
"long" (one comment per row). Otherwise it is "wide" (one column per question). Real examples:
the Coco file → "wide, open-ended columns: What did you LIKE about this product overall? / What did you NOT LIKE…";
the beverage file → "long, questions in `Source` = [Like, Dislike], text in `Text`".

**Cleaning the text.** Extra spaces are collapsed. A broken apostrophe from a bad export encoding ("It�s crunchy") is turned back into `'`, so words stay whole.

**Spelling repair** (`input.spell_correct`, on by default). Typos are repaired **in the text the models read**; the answer shown in your results stays as written.
Real examples: Coco `artifical → artificial`, `alway → always`; beverage `definetly → definitely`, `taaste → taste`, `flacour → flavour`.
A word is only changed when *all* of this is true, so brand names, slang and spellings people meant are left alone:
- the English dictionary does not know it (and neither its British/American twin: *colour/color* are both fine);
- it is rare in this survey (fewer than 3 uses), or another spelling is used at least 5 times as often;
- it has at least 5 letters, is not written with a capital letter in the middle of a sentence (a name or brand), and is not one of your protected theme words;
- a dictionary word starting with the same letter is one edit away (two for words of 7+ letters) and that word is one this survey already uses or a really common English word.
A missing apostrophe is always repaired because it changes meaning: `dont like it → don't like it` (the sentiment models read negation).
Every change is written to the `spelling` sheet.

**Non-answers.** Answers that say "nothing to report" must not be rated as opinions. An answer is marked not valid when it is empty or shorter than `min_chars`,
when it matches a pattern (`n/a`, `none`, `no comment`, `nothing …`), or when it **means** the same as an example non-answer (cosine at least 0.8 to examples such as
"I have no complaints", "Nothing at all", "Can't fault it"). Real examples removed in the Coco file: "I can't think of anything.", "Nothing I disliked.",
"Nothing, we all loved it.", "I struggle to find anything negative.", "I don't have any dislikes." In that file 109 of the 600 answers were non-answers, all of them in the "NOT LIKE" question (so 191 of its 300 answers are real complaints), and the
themes are built from the 491 real answers.

**Result:** a table with the original text, the cleaned text, a `valid` flag and a text key (a fingerprint of the cleaned text, so identical answers are processed once).

### 6.2 Segment ([segment.py](src/survey_nlp/segment.py))

**Goal:** one answer often says several things. Rating the whole answer would mix them, so the work is done on **clauses**.

**How:** first the text is split into sentences (a simple rule-based splitter). Then each sentence is split again at contrast words (`but, however, although, though, yet, whereas`)
and the punctuation around them is trimmed. A piece is kept if it has at least 2 words (`min_words`). A very short answer that would otherwise vanish is kept whole.

Real examples (Coco):

| Answer | Clauses |
|---|---|
| It's crunchy to begin with but nice and soft when chewed. | "It's crunchy to begin with" · "nice and soft when chewed" |
| It tastes as good as the original chocos but with a nice twist. | "It tastes as good as the original chocos" · "with a nice twist" |
| Strong chocolate flavour but not overpowering. they are crunchy and the pieces are a good size. and the aftertaste is nice | "Strong chocolate flavour" · "not overpowering" · "they are crunchy and the pieces are a good size" · "and the aftertaste is nice" |

Splitting happens on **unique** answers only (the 491 valid Coco answers, 488 different texts, gave 565 clauses). Identical clauses share one fingerprint, so each is processed once.

### 6.3 Embed ([embed.py](src/survey_nlp/embed.py), [cache.py](src/survey_nlp/cache.py))

**Goal:** a computer cannot compare "crunchy" and "crispy" as text, but it can compare numbers. The embedding model `BAAI/bge-small-en-v1.5` reads a clause and returns
384 numbers (a "unit vector") that capture its meaning. Clauses about the same thing get similar numbers.

**Similarity** between two clauses or between a clause and a theme is the cosine of their vectors. For all clauses against all themes this is a single matrix product, so it is fast.

**Cache:** vectors are stored on disk (`cache/BAAI__bge-small-en-v1.5.npz`) under a fingerprint of the text, so every unique text is embedded once, ever. The same cache holds the
vectors of example phrases (theme seeds, non-answer examples, name candidates, the sentiment axis). A second run on the same file takes seconds.

### 6.4 Themes ([themes.py](src/survey_nlp/themes.py))

**Goal:** find what people talk about, without being told. This is the largest module.

#### In simple words

Imagine every clause written on a **sticky note** (565 notes for Coco). Your job is to put the notes about the same thing into piles, without anyone telling you what the piles are.
The program does it in six easy moves:

1. **Ignore the mood.** "I love the packaging" and "the packaging is awful" are about the *same thing*, so the program first forgets whether a note is praise or complaint
   (and forgets that every note is about "this product"). Otherwise it would make a "praise" pile and a "complaint" pile instead of a "packaging" pile.
2. **Set aside the empty notes.** "I like it", "It is good", "Not for me" say nothing about a topic. They are put aside (they end up in the General comment theme later).
3. **Pick out the topic words on each note.** From "Strong chocolate flavour" keep *chocolate* and *flavour*. Throw away small words (*the, is, very*) and judging words (*good, nice*).
4. **Group the words, not the notes.** Words that are used in notes about the same thing are put together. *flavour, cocoa, aroma, taste* end up in one word-group; *crunchy, crunch, soft* in another;
   *packaging, label, bottle* in a third; *price, worth, cheap* in a fourth.
5. **Let each note vote.** A note joins the pile whose word-group has most of the note's words. A word that few notes use counts more than a common one, because a rare word is a stronger clue.
6. **Tidy up.** A pile that is far too big (more than 30% of all notes) is split into smaller piles. Tiny piles (fewer than 8 notes) are thrown away. If there are more than 10 piles, only the 10 biggest are kept
   and each smaller pile is added to the most similar big pile, so no note is lost. Finally each pile gets a name and some keywords (section 6.5).

Small illustration (clauses are real Coco clauses; the word-groups are simplified):

| Note (clause) | Topic words found | Word-group with most of them | Pile |
|---|---|---|---|
| Strong chocolate flavour | chocolate, flavour | flavour, cocoa, aroma, taste | **Flavour** |
| It's crunchy to begin with | crunchy | crunchy, crunch, soft | **Crunch** |
| Attractive packaging and appetite | packaging, appetite | packaging, label, bottle | **Packaging** |
| Don't like the shape of the pieces much | shape, pieces | pieces, shape, size | **Pieces** |
| I like it | (none: set aside in move 2) | | **General comment** |

**Why words, and not simply "group similar sentences" (UMAP + HDBSCAN, method `density`)?** These are design reasons, not a measured quality win (see the table below):
- **A pile can be explained.** A word-based pile *is* its words ("packaging, label, bottle"). A density cluster is a thin region of meaning space that has to be described afterwards.
- **No random projection.** UMAP shrinks the meaning vectors with a random-                       seeded step; the same seed repeats, but a small change in the data or the library version can move the clusters. The word method has no random step at all.
- **Every note is used.** HDBSCAN labels clauses in thin regions as "noise" and leaves them without a pile; the word method gives every clause with a topic word a pile.
- **Fewer fragile knobs.** HDBSCAN needs a minimum cluster size and similar settings that depend on how many answers there are.

**But it is not clearly more accurate.** Same datasets, same settings, only the method changed:

| mean over the 3 dev datasets | words (default) | density (UMAP + HDBSCAN) |
|---|---|---|
| mixed themes | 1.00 | 1.00 |
| name coverage, lowest / mean | 0.277 / 0.492 | 0.251 / 0.509 |
| largest-theme share | 0.261 | **0.218** |
| Gini of theme sizes (0 = even) | 0.323 | **0.226** |
| General comment share | 0.071 | 0.052 |

| held-out app reviews | words | density |
|---|---|---|
| mixed themes | 3 | 2 |
| name coverage, lowest / mean | 0.192 / 0.292 | 0.210 / 0.307 |
| largest-theme share | **0.126** | 0.204 |
| Gini | **0.076** | 0.194 |

So density gives more even themes on the three dev datasets, the word method gives more even themes on the held-out set, and name coverage is similar. There is no clear winner; the word method was kept as the default
for the reasons above and because it was chosen earlier in the project. Changing the default would need more datasets to decide.

**What can go wrong:** a pile can mix two topics that people mention together ("pieces" and "price" in one answer), or follow the wording of repetitive text. The program notices piles whose name fits too few
of their notes and flags or splits them (section 6.5).

#### The same steps in technical detail

**a) Centring.** All clauses in a survey share a common "flavour" (they are all about *this product*, and many start with "I like…"). That shared direction is partly subtracted
(`themes.center_strength` 0.7: 70% removed) so that differences between topics stand out.

**b) Taking tone out.** Clustering on raw meaning tends to group "love the packaging" with "love the taste" (both are praise) and to split "the packaging is nice" from "the packaging is bad".
A positive-versus-negative direction is estimated from example phrases (`themes.sentiment_axis`: "It is good, I love it…" vs "It is bad, I hate it…") and removed before grouping.
Result: themes form by **topic**, not by tone.

**c) Setting generic remarks aside.** Clauses very close to the General comment seeds ("It is good", "I like it", "Not for me") say nothing about a topic. Those with similarity
at least `general_exclude_sim` (0.4) are left out of the grouping. They still get a theme later (usually General comment).

**d) Finding candidate themes** (`themes.method: aspects`, the default). The method works from **words**, not from blobs of clauses, and has no random step, so the same data always gives
the same themes.
1. The topic words of every clause are collected: nouns, names and plain describing words ("sweet", "crunchy"); function words, judging words and fillers are dropped, and plurals are
   folded ("pieces" = "piece"). A word must occur in at least 4 clauses and in at least 1% of all clauses (whichever is more).
2. A word is represented by the average meaning of the clauses that use it. So "packaging", "label" and "bottle" end up close together; "price" and "worth" end up close together.
3. The words are grouped (average-linkage clustering) into about `n_themes + 2` groups.
4. Each clause goes to the group that **most of its words belong to** (rarer words count more).
5. A group holding more than `split_large_frac` (30%) of all clauses is split in two with ward linkage, up to three rounds. Groups with fewer than `min_theme_clauses` (8) are dropped.

An alternative method `density` (UMAP + HDBSCAN on whole-clause meaning) exists and is used when the word method finds fewer than `min_themes` groups.

**e) Reducing to N themes.** The `n_themes` largest groups are kept (default 10). Each smaller group is folded into the most similar kept group, so its clauses are not lost.

**f) Naming and keywords** for every group: see 6.5.

**g) Folding generic groups.** A group with no informative word of its own, or whose clauses sit close to the General comment seeds (`generic_theme_sim` 0.4), is not a topic ("easy drink,
overall, balanced"). It is dropped and its clauses are assigned later, where most end up in General comment. Example: a group named "Tea" with no informative words was folded away.

**h) Your own themes and edits.** `themes.required` themes (names with optional keywords and example phrases) are added as themes of their own; a required theme's centre is built from its example phrases and
pulled towards the clauses that mean the same thing. General comment is a required theme with `general: true`. `themes.curation` can merge, rename, drop, add or sharpen themes by name. These edits come **after**
discovery, so they never change which topics were found.

**i) Refinement.** Clusters only contain the dense core of each topic. Each theme centre is pulled towards the clauses that are now nearest to it (`refine_iters` 2) so borderline clauses are
matched, then the sizes are recounted. `n_total` can fix the total number of themes: if you add a theme of your own the smallest discovered theme is dropped to stay at the total.

**What you get for Coco (real, 10 themes + General comment):**

| Theme | Clauses | Main words (keywords) | Example clause |
|---|---|---|---|
| Taste | 70 | taste, crispy | "sweet taste" |
| Crunch | 89 | crunchy texture, flavour | "nice and crunchy" |
| Flavour | 74 | flavour, cocoa taste, rich chocolate, flavor, aroma | "The flavour is not very strong" |
| White Chocolate | 55 | white chocolate, mixture | "Too much white chocolate flavour" |
| Pieces | 45 | pieces, colours, shape, price, size | "Don't like the shape of the pieces much" |
| Milk | 20 | milk, bowl, pieces | "It went a little soft in milk quicker than I would have liked" |
| Packaging | 25 | packaging | "Attractive packaging and appetite" |
| General comment | 36 | (none) | "I like it" |

Sizes are those found at discovery; the final counts after assignment are a little different (see 6.7).

### 6.5 Names, main words, keywords and examples

**Main words.** For each group the words that are typical of **that group more than of the others** are ranked (class-based TF-IDF). Then they are cleaned:
- **Filler and judging words** are removed so "good", "really", "nice" never become a theme's name or keyword. By default (`themes.filler_mode: language`) this uses a short list of general English
  judging and filler words. It contains **no topic words** (the entries "drink", "meal", "buy", "day", "people" that were once added while tuning on food surveys are gone).
  Optional modes find judging words from the data (words lying far along the positive/negative axis) instead; see section 13 for why that is not the default.
- **Verbs and fragments** are removed. The part of speech is learned from how the word is used in this survey ("lingers" in "the aftertaste lingers" is a verb; "packaging" is a noun).
- **Words used everywhere** are removed (no lift): a word the theme uses barely more than the rest of the survey describes nothing about it.
- **Repeats** are merged: "drink", "easy drink", "drink overall" keep one entry; a phrase replaces its word when it covers most of the word's clauses.

**Choosing the name.** Among the cleaned candidates (nouns or plain describing words, used by a fair share of the theme's clauses) each is scored by
closeness in meaning to the theme centre, the share of the theme's clauses that use it, and how much more the theme uses it than the survey. A phrase beats its own word when it covers most of the word's
clauses ("White Chocolate" beats "Chocolate"). Penalties apply for describing words ("Easy" is a worse name than "Packaging"), for evaluation words, and for a name that means the same as a name already taken ("Taste" next to "Flavour").

**Checking the name against its members** (`themes.name_min_coverage`, 0.25). After naming, the name is measured: what share of the theme's clauses contain it? (A form such as "crunchy" counts for "Crunch"; "pieces" for "piece".)
- Below the bar, the **next-ranked candidate** is tried (never an evaluation word or a repeat of another theme's name).
- If none passes, the theme is **mixed**. It is then **split in two** (ward linkage on its clause vectors, no random step), and the split is kept only when both halves are big enough
  (`min_theme_clauses`), both get a passing name, and no other theme becomes mixed.
- The clusters are only the dense cores, so the name is **checked a second time** against the clauses the theme is expected to show once everything is assigned. A failing name is replaced or the theme is marked mixed.
  This second check runs before your edits, so edits you write refer to the final names.

Real example: in Coco a theme first called "Cold Milk" was used by only 15% of the clauses it showed. The next candidate, "Chocolate", was used by 72%, so it was renamed.
Mixed themes carry `name_mixed` in `discovered_themes.json` and are marked on the `quality` and `themes` sheets. Naming a theme yourself clears the mark.

**Keywords** (`themes.n_phrases`, default 6). Candidates are the **noun chunks** and every run of **two or three content words** found in the theme's own clauses
("cocoa taste", "rich chocolate", "clear ingredient list"), minus filler, verbs and words with no lift. They are ranked by closeness in meaning to the theme centre plus the lift. A phrase that is
a substring, plural form or extension of a better one is dropped ("sweet", "too sweet", "sweetness" keep one). The final six are chosen with **MMR** (maximal marginal relevance): each next phrase is the best trade-off
between being relevant and being unlike what is already chosen, so the list is not six near-copies. Real result for Coco, theme Flavour: *flavour, cocoa taste, rich chocolate, flavor, aroma*.
A theme with little vocabulary gets fewer phrases (Packaging gets just "packaging").

**Example clauses** (`themes.n_examples`, default 3): the three clauses nearest the theme centre (each text once), shown as `example_1..3` on the `themes` sheet, so you can see what a theme is about without reading hundreds of answers.

**Optional:** `themes.aspect_labels` (empty by default). If you supply generic labels (for example Taste, Price, Packaging), a theme whose best name is only a describing word ("Greasy") or not a noun gets the closest label when its centre is at
least `aspect_label_min_sim` (0.45) close to it. No labels are built in.

### 6.6 Saved clusters and stability ([themes.py](src/survey_nlp/themes.py), `clusters.json`)

The groups of clauses found in 6.4 are saved in `<run_dir>/clusters.json`, keyed by a fingerprint of: the clauses, the embedding model, the seed, the method, the discovery settings, the word lists used for filler, and the General comment seeds.
Anything that changes *which topics exist* changes the fingerprint and the groups are found again. Anything that only changes what happens **after** discovery
(naming, your required themes, edits, the number of keywords…) does not.

Result: **renaming, merging and adding themes does not reshuffle the others.** Checked on the beverage data with 11 themes:

| You do | What happens to the other themes |
|---|---|
| rename "Flavour" → "Taste" | nothing: every other theme keeps its name and size exactly |
| merge "Orange" + "Sweet" | the others keep their names; sizes move by a few clauses because the merged theme's centre takes clauses |
| add a theme "Price" (total + 1) | all discovered themes survive with the same names; sizes change by 0 to 1 clause |
| add a theme while keeping the total fixed | the **smallest** discovered theme is dropped to make room (the total is what you chose) |
| merge two themes while keeping the total fixed | a previously cut theme comes back to fill the free slot |

Running the same data and settings twice gives identical themes. `--rediscover` forces a new search; `themes.freeze: false` turns saving off.

### 6.7 Assign ([assign.py](src/survey_nlp/assign.py))

**Goal:** for every clause, decide which theme(s) it talks about. There are three routes, and a clause can be matched by more than one:

| Route (`via`) | Rule | Real example (Coco) |
|---|---|---|
| **semantic** | the clause's meaning is at least `assign.min_sim` (0.4) close to the theme's centre; a second theme only if almost as close (within `secondary_margin` 0.1) | "It's crunchy to begin with" → Crunch (0.61) |
| **keyword** | the clause contains a theme keyword as a whole word. Words **you wrote** always match. Words taken from a theme's name only count if the clause is also at least `keyword_min_sim` (0.25) close | "I liked the mix of white and chocolate pieces" → Pieces (0.28, by the word "pieces") |
| **fallback** | a clause nothing else matched joins its closest specific theme if at least `fallback_min_sim` (0.25) close | "I get a good portion for the price point" → Pieces (0.36) |

When both semantic and keyword agree the route is shown as `both`. A real clause can get two themes:
"they are crunchy and the pieces are a good size" → **Crunch** (0.45) and **Pieces** (0.47).

Then: at most `max_themes_per_clause` (2) themes are kept per clause (the strongest; words you wrote are exempt). **General comment** only receives clauses no specific theme wanted, and it is dropped for an answer that already has a specific theme.
So "not overpowering" in the answer above gets no theme of its own, because the same answer already matched Flavour, Crunch, Pieces and Taste.

For Coco: 553 of the 565 clauses got a theme (383 have one theme, 170 have two), which makes 723 clause-theme matches: 334 semantic, 247 both, 86 keyword, 56 fallback.

### 6.8 Sentiment of the whole answer ([sentiment.py](src/survey_nlp/sentiment.py))

The model `cardiffnlp/twitter-roberta-base-sentiment-latest` reads the whole answer and returns three probabilities (negative, neutral, positive); the largest wins. This is the **Overall** column.
"Smell bad but taste ok" → Negative.

### 6.9 Sentiment per theme ([sentiment.py](src/survey_nlp/sentiment.py))

**Aspect-based sentiment.** The model `yangheng/deberta-v3-base-absa-v1.1` is given a clause **and the theme name** ("Smell bad" + aspect "smell") and says how that clause feels *about that theme*.
Every unique (clause, theme) pair is scored once; batches are sorted by length for speed.

**Making one rating per (answer, theme).** An answer can have several clauses about the same theme. For each (answer, theme) the largest positive and the largest negative probability across its clauses are taken, then:

- **Positive** if p_pos ≥ `evidence_threshold` (0.6) and p_neg is not larger than p_pos + `allowance` (0.05);
- **Negative** the same way round;
- if both qualify (a mixed opinion), `mixed_policy: stronger` keeps the stronger one;
- if neither qualifies the rating is **Neutral**.

These two numbers are applied at export, so changing them does not run the models again.

Real numbers for "Smell bad but taste ok":

| Clause | Theme (similarity, route) | p_neg | p_neu | p_pos | Model says |
|---|---|---|---|---|---|
| Smell bad | Smell (0.73, both) | 1.00 | 0.00 | 0.00 | Negative |
| taste ok | Flavour (0.56, semantic) | 0.01 | 0.82 | 0.17 | Neutral |

### 6.10 Rules on top of the models (`mode: extended`, the default)

Language models make typical mistakes. A few transparent rules correct them (`mode: spec` switches all rules off and keeps the models' ratings only).

1. **Negation fix.** The program learns which words are clearly positive or negative from the survey itself (a word must be rated the same way in every question where it appears). A negated bad word flips to positive:
   "not so artificial" → Positive; a negated good word flips to negative: "not good" → Negative. It does not fire when the same text also has a plain word pointing the other way.
2. **Question polarity.** A "what did you like" question asks for praise; a "what did you not like" question asks for complaints. The polarity is read from the answers (if praise outweighs complaints 3 to 1 it is a praise question), with the question's wording as a tie-break.
   In a praise question, merely *mentioning* a theme counts as praise, so a Neutral rating becomes Positive. In a complaint question it becomes Negative. A clearly evaluative opposite word ("smell **bad**" in a like question), a complaint phrase or a very confident model still wins.
   In the example: "taste ok" came back Neutral (0.82); in the Like question that becomes **Positive**; "Smell bad" stays **Negative**. Final: Flavour Positive, Smell Negative.
3. **Complaint phrases.** Wording that is a complaint by itself ("too sweet", "not fizzy enough", "lacks flavour", "a bit sweet" with nothing good in the answer) counts as Negative in a praise question even when the model is unsure.
   The patterns are regular expressions in the config, not in the code.
4. **Lukewarm answers.** "Okay", "nothing special", "average" make the answer Neutral, unless a clearly positive or negative word is also present ("okay but awful smell" is not lukewarm).

### 6.11 Export ([export.py](src/survey_nlp/export.py))

Writes the workbook of section 3: short readable question names, coloured cells, a frozen header, a `Combined` sheet when there is more than one question, the `themes` sheet (with the example-clause columns), the `quality` sheet and `quality.json`.
If the old workbook is open in Excel (Windows locks it) the file is saved under a new name and a warning is printed.

---

## 7. One answer from start to finish

Real answer from the beverage survey (question "Like"): **"Smell bad but taste ok"**

1. **Ingest:** a valid answer, no typos to repair, not a non-answer.
2. **Segment:** the contrast word "but" splits it: **"Smell bad"** · **"taste ok"**.
3. **Embed:** each clause becomes 384 numbers.
4. **Themes:** (found once for the whole survey) among them a theme about smell and the theme **Flavour**.
5. **Assign:** "Smell bad" is 0.73 close to the smell theme (semantic and keyword agree). "taste ok" is 0.56 close to Flavour (semantic).
6. **Overall:** the whole answer is rated Negative.
7. **Per theme:** "Smell bad" about smell: negative 1.00 → Negative. "taste ok" about Flavour: neutral 0.82 → Neutral.
8. **Rules:** it is a Like question, so a bare mention counts as praise: Flavour → **Positive**. "bad" is a clearly negative word, so Smell stays **Negative**.
9. **Export:** the row reads *Overall Negative · Flavour Positive · Smell Negative*, coloured red / green / red in the workbook, with the probabilities on the `detail` sheet and the clause matches on the `pairs` sheet.

Real answer from the Coco survey: **"Strong chocolate flavour but not overpowering. they are crunchy and the pieces are a good size. and the aftertaste is nice"**
is cut into 4 clauses: "Strong chocolate flavour" → Flavour (0.61) · "not overpowering" → no theme of its own · "they are crunchy and the pieces are a good size" → Crunch (0.45) and Pieces (0.47) · "and the aftertaste is nice" → Taste (0.40).

---

## 8. The quality report

A weak run should be visible without reading every answer. Every export writes the `quality` sheet and `quality.json`, and prints one summary line on the command line (the web app keeps its log short).

**Per theme:**

| Measure | Meaning | Example |
|---|---|---|
| name coverage | share of the theme's clauses that contain its name | Chocolate: 0.71; a theme named "Greasy" in the synthetic set: 0.68 |
| name lift | how much more the theme uses its name than the rest of the survey | 6 = six times as often |
| coherence | do the theme's main words appear together in the same answers (mean NPMI, from −1 to 1)? | higher is better; see the note below |
| size and share | clauses in the theme and their share of all clauses | Flavour: 21% |
| adjective name | the name is only a describing word (judged from how the answers use it: after "too", "very", "is"…) | "Crispy" |
| mixed | the name fits too few clauses (see 6.5) | |

**Per run:** the smallest and mean name coverage; the number of themes below 0.25 (a fixed yardstick, `quality_bar`, so runs stay comparable); the number of weak names; the number of mixed themes; coherence;
the largest-theme share and the **Gini** of the sizes (0 = all themes equal, towards 1 = one theme holds everything); the General comment share; the share of clauses with no theme; the number of adjective names; and the
number of pairs of themes that look like duplicates (centre cosine ≥ 0.85 or main-word overlap ≥ 0.4). Your own themes and General comment are never counted as weak or duplicate.

Real run summary (Coco): 10 themes; name coverage lowest 0.34, mean 0.62; largest theme Flavour 21%; Gini 0.28; General comment 7%; 2% of clauses without a theme; 1 mixed theme (Pieces).

**Note on coherence.** The keywords are now varied phrases on purpose (MMR). Varied phrases appear together less often than near-copies, so this number looks worse than it used to even though the keyword lists read better.
Judge the keywords by reading them; use coherence to compare runs with the same keyword settings.

---

## 9. Caching and stage keys

Each stage has a key made from: the input file (size and modification time), the keys of the stages before it, the config sections it depends on, and a **fingerprint of the code** the stage runs (so results made by older code are never reused). The dependencies are in `PLAN` in
[pipeline.py](src/survey_nlp/pipeline.py).

Examples:
- change only `output.labels` or `output.codes` → only the export runs again;
- change `absa.evidence_threshold` → nothing runs again except the export;
- change a `themes` setting → themes, assign, absa and export run again, but ingest, segment, embed and the sentiment of whole answers do not;
- edit `themes.py` → themes onward run again (the fingerprint changed).

Typical timings on a laptop CPU for the Coco file (about 565 clauses): about a minute cold, about 1.4 seconds when nothing changed.

---

## 10. The web app (`streamlit run app.py`)

Three tabs:

1. **Data.** Upload a file or pick one from `data/`; the program shows the detected layout and open-ended columns and lets you correct them.
2. **Themes and analysis.** Choose how many themes (slider or box). **Find themes** runs ingest to assign and fills an editable table: *Theme, Keep, Main words, Answers* (the number of different answers assigned). You can:
   - **rename** (edit the name in the table) or **remove** (untick Keep), then press *Apply table changes*;
   - **merge** two or more themes into one (give it a name or keep the first);
   - **add** a theme of your own with optional signal words.
   Progress of an edit shows under the three buttons; the run log shows only the short stage lines. Edits are listed under "Edits so far"; "Undo all my edits" resets. **Run the analysis** runs the sentiment stages and exports.
3. **Results.** Headline numbers, one tab per question with a bar chart of how each theme is rated, a table you can filter by theme and search by text, and the download of the workbook.

The app uses `config_default.yaml`, so all settings in section 11 apply. Because edits come after discovery (6.6), the other themes stay stable. If the table on screen is older than the settings in force (a changed number of themes, a new version of the code) the app says so and asks you to press **Find themes** again.
The `quality` sheet is hidden from the question tabs (it is an internal sheet).

---

## 11. Settings (most useful ones)

Everything is in a YAML file; `config_default.yaml` works on any new survey. "off" means the line is not in the file.

| Section.key | Default | Meaning |
|---|---|---|
| `mode` | extended | `spec` = models only, `extended` = rules on top |
| `seed` | 42 | seed for the density method and sampling |
| `input.spell_correct` | true | repair typos |
| `input.non_answer_min_sim` | 0.8 | meaning match for non-answers |
| `segment.contrast_words` | but, however, although, though, yet, whereas | where a clause is split |
| `segment.min_words` | 2 | shortest clause kept |
| `embed.model` | BAAI/bge-small-en-v1.5 | sentence embedding model |
| `themes.method` | aspects | `aspects` (word based, no random step) or `density` |
| `themes.n_themes` | 10 | themes to keep (General comment is added on top) |
| `themes.split_large_frac` | 0.3 | a group above this share of clauses is split in two |
| `themes.min_theme_clauses` | 8 | smallest theme |
| `themes.center_strength` | 0.7 | how much of the shared direction is removed |
| `themes.general_exclude_sim` | 0.4 | clauses this close to General are not clustered |
| `themes.generic_theme_sim` | 0.4 | a group this close to General is folded into it |
| `themes.required` | General comment | themes that must exist (names, keywords, seeds) |
| `themes.name_min_coverage` | 0.25 | share of a theme's clauses that must use its name; below it the next candidate is tried, none fits = mixed (then split in two if both halves pass). 0 turns the check off |
| `themes.name_combine` | false | allow two-part names ("A & B") |
| `themes.n_phrases`, `phrase_mmr_lambda`, `n_examples` | 6, 0.6, 3 | keywords per theme, how varied they are (1 = only the most relevant), example clauses per theme. No line = older word list / no examples |
| `themes.filler_mode`, `filler_k` | language, 2.0 | which words never name a theme. `language` = general English judging / filler words only; `fixed` = the original longer list; `data` / `hybrid` = judging words found in the survey (more than `filler_k` standard deviations along the positive/negative axis). See section 13 |
| `themes.aspect_labels`, `aspect_label_min_sim` | `[]` (off), 0.45 | your own generic labels for themes named after a describing word |
| `themes.merge_k`, `split_ratio`, `split_k`, `split_min_gain`, `name_min_lift` | off | optional relative rules (merge outlier pairs, split clearly separable big themes, judge names by lift). Measured worse, see section 13 |
| `themes.merge_sim`, `merge_jaccard`, `merge_name_sim` | off | older absolute merge rules; they still work when set |
| `themes.freeze` | true | reuse saved clusters |
| `assign.min_sim` | 0.4 | semantic entry bar |
| `assign.secondary_margin` | 0.1 | how close a second theme must be |
| `assign.keyword_min_sim` | 0.25 | closeness needed for a name-word match |
| `assign.fallback_min_sim` | 0.25 | closeness for the leftover rule |
| `assign.max_themes_per_clause` | 2 | cap per clause |
| `absa.evidence_threshold`, `allowance` | 0.6, 0.05 | sentiment decision |
| `output.codes`, `labels`, `mixed_policy` | 0/1/2, words, stronger | what is written |
| `output.auto_polarity` | true | read from the answers whether a question asks for praise or complaints |

---

## 12. Tests and evaluation

- `pytest -q` runs 96 tests: ingest and layout detection, spelling, segmentation, the cache, the command-line edits, theme discovery, naming, name checks, keywords, assignment rules, sentiment rules, the quality report,
  the evaluation suite helpers, and guards for the web app (it must be valid Python; sheet names the export writes must be hidden from the question tabs; the progress and log settings).
- **Evaluation suite** (`eval/run_suite.py`, datasets listed in `eval/suite/datasets.json`): five datasets run through the whole theme pipeline and scored with the quality report:
  Coco Pops, the beverage answers with the fixed-codebook config and with the generic config, a synthetic snack/drink set (templated, repetitive wording), and `app_reviews`, a generated non-food (service/app-review) set.
  `app_reviews` is **held out**: it is reported on its own and was never used to choose a default or a threshold.
  `python eval/run_suite.py --compare eval/baseline_suite.json` prints per-dataset tables, a mean over the dev datasets and the held-out row. `--set themes.key=value --tag _x` runs an experiment in its own folders.
  `--legacy` switches the new theme-quality settings off (the behaviour before that work); `eval/baseline_suite.json` was made that way and reproduces the earlier baseline `eval/baseline_quality.json` on the three older datasets.
  The real surveys stay in `data/` (not committed); the synthetic files are in `eval/suite/` and reproducible with `eval/suite/make_app_reviews.py`.
- Older calibration scripts: `make_label_sheet.py`, `evaluate.py`, `make_review_sheet.py`, `theme_stability.py`, `compare_to_reference.py`.
- **Caveat:** the labelled samples used for calibration (about 200 pairs each) were written by the coding assistant, not by a human. The quality numbers are proxies (name coverage, word co-occurrence, size balance), not a verdict on whether the themes are "right".

---

## 13. What was tried and what is on or off

The aim was themes that are well named, balanced and not made of generic words, on **any** survey. Every change was measured on the suite; a change that helped one dataset but hurt others is not on by default.

**Combined effect of the shipped defaults (mean over the three dev datasets; held-out in brackets), compared with the behaviour before this work:**

| | before | now |
|---|---|---|
| themes with a name used by under 25% of their clauses | 1.67 (4) | 0 (0) |
| themes marked mixed (flagged, not fixed) | 0 (0) | 1.0 (3) |
| lowest name coverage | 0.155 (0.184) | 0.277 (0.192) |
| mean name coverage | 0.442 (0.337) | 0.492 (0.292) |
| largest-theme share | 0.272 (0.214) | 0.261 (0.126) |
| Gini of theme sizes | 0.324 (0.226) | 0.323 (0.076) |
| coherence (see the note in section 8) | −0.096 (−0.285) | −0.291 (−0.602) |

Much of the drop in "below 25%" is themes now being flagged mixed instead; the better name checks are the main real gain (for example "Cold Milk" at 0.15 became "Chocolate" at 0.71).

**Theme-finding method.** `aspects` (words, default) versus `density` (UMAP + HDBSCAN): no clear winner; see the comparison tables in section 6.4.

**On by default:** name check with fallback and split (6.5), second check against the clauses the theme shows, keyword phrases with MMR, example clauses, filler list without topic words (`language`).

**Off by default, and why (all measured on the suite):**

| Option | What it does | Why it is off |
|---|---|---|
| `filler_mode: data` / `hybrid` | find judging words from the data only | finds the right words ("nice", "unpleasant") but without the list generic nouns form themes ("Greasy" with 282 of 864 clauses; "Time" with 139 of the app reviews); worse balance on every dev set |
| `merge_k` (outlier pairs) | merge theme pairs that stand out in similarity | some pair always stands out; it merged different topics (Notifications + Adverts) and made blobs |
| old absolute merge rules | merge at fixed similarity / word overlap | with the fixed numbers it merged three pairs on one dataset; removing them improved balance and coverage on the dev sets |
| `split_ratio`… (relative split) | split a theme much bigger than the median when it separates clearly | never fired on these datasets |
| `name_min_lift` | judge a name by its lift over the rest of the survey | removes the "mixed" flags by letting weaker names pass (it brought "Cold Milk" back); better used to rank names than to accept or reject |
| `aspect_labels` | map a describing-word name to your own generic label | none built in; closest label to a small specific theme is rarely close enough |

---

## 14. Known limitations and open problems

- English only; at least a few hundred answers per question are needed for stable themes.
- Themes from clustering are not guaranteed to be clean topics. On repetitive or templated text the groups follow the sentence wording. Examples on the synthetic set: "Texture" (33% of clauses) and "Flavour" stay flagged mixed; a theme about greasiness is named "Greasy" or "Mouth".
- A name can still be a weak choice that passes the 25% bar (for example "Mouth"). Choosing the name from the members with a coverage × lift × similarity score is the next planned step (section 15).
- A few weak words remain in keyword lists ("premium", "price", "idea") because the filler list is short on purpose and the data-derived filter is not reliable enough yet.
- The coherence number penalises varied keywords (section 8).
- A fallback match can be loose: "I get a good portion for the price point" went to Pieces at 0.36.
- A clause can end up with no theme because a specific theme already matched another clause of the same answer ("not overpowering" above); about 2% of Coco clauses.
- A negation still fools the sentiment model in some phrasings ("isn't as sweet as I thought").
- `config.yaml` (hand-curated for the Coco survey) stops on a fresh run: its `curation.seeds` names a theme "Flavour" that the current word-based discovery does not always produce. Use `config_default.yaml`.
- The calibration labels are not human labels (section 12).
- If the optional merge rules are switched on, they skip any theme named in your required themes or edits, so an edit could change what is merged. They are off by default, so this does not happen normally.

---

## 15. What comes next

1. **Name from members (planned).** Choose each name as the head noun phrase with the best *coverage × lift × similarity* among the theme's own clauses, so "Fizzy" loses to "Cola" when most clauses say cola; keep the mixed flag and split.
2. **Self-validation flags in the app (planned).** Show mixed / low coverage / possible duplicate in the app's theme table (a "Flag" column).
3. **Phrases for small themes (planned).** Fall back to the top lifted words when a theme has few candidate phrases.
4. **Auto-tune (planned).** A small deterministic grid over the number of themes and the split setting, scored by a quality score (name coverage, coherence, balance, General comment share), with the chosen values logged in `quality.json`.
5. Replace the short filler list with a fully data-driven filter that does not let generic nouns form themes.
6. Re-label the calibration sample by hand and re-tune `assign.min_sim`, `keyword_min_sim` and `fallback_min_sim` on it.
