# Survey NLP: open-ended answers to themes and per-theme sentiment

Built from `PROJECT_SPEC.md`. Input: `data/coco_pops_VR.xlsx` (Coco Pops VR tasting survey, 300 respondents).
Open-ended questions analysed: "What did you LIKE about this product overall?" (Likes) and
"What did you NOT LIKE about this product overall?" (Dislikes).

Run: `survey-nlp run` (use `--until themes` to inspect themes, `--force` to ignore checkpoints).
Output: `runs/latest/aspect_sentiment_results.xlsx` (sheet `results` = Positive/Negative/Neutral words,
`results_codes` = 0/1/2, plus `detail`, `pairs`, `themes`, `timings`).

## Timings (CPU, full run with `--force`, embedding cache warm)

| stage | seconds |
|---|---|
| ingest | 0.54 |
| segment | 0.25 |
| embed | 0.00 (cached; first-ever run about 25) |
| themes | 14.81 |
| assign | 0.05 |
| overall | 12.75 |
| absa | 34.08 |
| export | 1.21 |
| **total** | **63.7** |

A re-run with nothing changed takes about 1.4s. Changing only `absa.evidence_threshold`, `absa.allowance` or
`output.*` re-runs only export.

## Calibration (200 labeled pairs)

**Caveat: the labels were written by Claude (the coding assistant), not by a human annotator**, and without
looking at the model scores. Treat the numbers as a development estimate. Re-label by hand before reporting them externally.
The sample has few neutral cases (7), so neutral-class metrics are unreliable.

- Relevance (semantic score only), best `assign.min_sim` = 0.775: P 0.77, R 0.89, F1 0.82
  (0.80 gives P 0.80, R 0.81, F1 0.81). The spec's starting guess of 0.50 never rejects anything with bge-small
  (median top-1 cosine 0.83).
- ABSA on relevant assigned pairs (n=107): macro-F1 (argmax) 0.696; at `evidence_threshold` 0.60, `allowance` 0.05: 0.690.
  Sweeping `evidence_threshold` 0.40-0.90 gives 0.63-0.70 and `allowance` 0-0.20 changes nothing. This is within noise
  for 107 pairs, so the spec values (0.60 / 0.05) were kept.
- Theme stability (5 subsamples at 80%): mean matched cosine 0.91-0.97, 8-10 themes.

## Deviations from the spec (all deliberate)

- `input.header_row: 3`, `input.skip_rows: 1`, explicit `text_cols` (the export has a 4-row preamble and a question-ID row).
- Wider `input.non_answer_regex` ("nothing disliked", "no downsides" style answers are not themes).
- `themes`: `min_cluster_size` 15 -> 8, `min_samples` 5 -> 3, `merge_cosine` 0.85 -> 0.90, `min_theme_clauses` 20 -> 12
  (the corpus has only about 600 clauses).
- `assign.min_sim` 0.50 -> 0.775 (calibrated); new `assign.keyword_min_precision: 0.5` drops generic keywords.
- ABSA cache key excludes `evidence_threshold` and `allowance` (applied at export), so tuning them does not re-run the model.
- `output.labels` / `output.question_labels`: words and short headers in `results`; 0/1/2 kept on `results_codes`.
- `eval/evaluate.py`: `min_sim` sweep extended to 0.95.
- Model names unchanged.

## Update: quality improvements (after reviewing the first output)

Problems found on real data and what was done (all in `config.yaml`; defaults reproduce the original spec behaviour):

| Problem | Cause | Fix |
|---|---|---|
| "I like the monkey" scored as Flavour | Every clause was 0.6-0.9 cosine to every theme centroid (shared "I like it" wording), so themes barely separated | `themes.center` / `assign.center`: embeddings are centred before centroids and scoring. Max centroid cosine fell from 0.90 to 0.51 |
| "I didn't find any negative", "Nothing to report on the negative side." scored Negative | The sentiment model reads the word "negative"; regexes can't keep up with phrasings | `input.non_answer_examples` + `non_answer_min_sim: 0.80`: answers whose meaning matches example phrases are skipped (109 skipped). The regex is kept as a first filter |
| Milk and Cold Milk separate; unstable discovery (Colour vanished once non-answers were removed) | Unsupervised clustering of about 600 clauses is unstable; raw similarity merging would chain Flavour into everything | `themes.curation`: `merge`, `rename`, `drop`, `extra_keywords`, `seeds`, `add` (hand-defined theme from example phrases, used for Colour) |

Final themes: Flavour (merged with Taste), White Chocolate, Milk, Pieces, Packaging, Coco Pops Brand, Aroma, Sweetness, Colour.
"Bad Points" is gone because those answers are now skipped as non-answers.

Re-calibration on a second 194-pair labeled sheet (again labeled by Claude, same caveat as above):
- `assign.min_sim` 0.45 (centred space). Whole-assignment F1 about 0.85 (0.40-0.60 is a plateau; keywords carry most of it).
- ABSA macro-F1 0.728 on 99 relevant pairs with `deberta-v3-base-absa-v1.1`. Tried `deberta-v3-large-absa-v1.1`: 0.667 and about 30x slower, so not adopted. Model names unchanged.
- `evidence_threshold` 0.60 and `allowance` 0.05 kept (differences within noise).
- Full run on CPU: 54.8s.

Review sheet: `python eval/make_review_sheet.py [filename]` -> one sheet per question with the answer, Overall, theme columns and the clauses used.

## Update 2: feedback round (review of the Likes sheet)

- Hand-written keywords for a theme (`themes.curation.extra_keywords`) now REPLACE the discovered ones (generic words such as "good", "little" were
  causing wrong themes). A keyword starting with `re:` is a regex, e.g. `re:milk(?! chocolate)`.
- New theme "Price"; more vocabulary for Flavour, White Chocolate (vanilla), Pieces (shape, size), Packaging (packing).
- `output.question_polarity`: in the Likes question a neutral mention counts as Positive and a weak Negative overall (p < 0.80) becomes Positive.
  `detail` keeps the raw model result in `model_code`. Dislikes is NOT enabled (not reviewed yet); add `Dislikes: {neutral_as: negative}` to try it.
- Known remaining error: negation ("isn't as sweet as I thought", "aren't too sweet") still fools the ABSA model at theme level.

## Choosing the themes at run time

`survey-nlp run` asks: "Required themes" (comma-separated) and, per theme, optional signal words. These themes are guaranteed in the output;
themes discovered in the data are added on top. A required theme that matches a discovered one by meaning takes it over (renamed), otherwise it is added.
Skip the prompt with `--themes "Packaging,Price"` or `--no-prompt` (uses `themes.required` from the config). Last answers are remembered in
`runs/latest/required_themes.json`.

## Using it on another survey (what is generic and what is not)

Generic: ingest (xlsx/csv, any number of open-ended columns, column auto-detection that skips IDs, JSON and near-empty columns), segmentation,
embeddings, theme discovery, required themes, assignment, sentiment, export, caching.

Per survey you must set in `config.yaml`: `input.path`, `header_row`/`skip_rows` if the file has a preamble, `text_cols` (or `auto`),
`id_col`/`keep_cols`, `output.question_labels`, and optionally `output.question_polarity`.
Re-do or remove: `themes.curation` (theme names are specific to this survey; an unknown name raises an error that lists the real ones),
and re-check `assign.min_sim` and `input.non_answer_min_sim` on a labeled sample (`eval/`).

Limits: English only (models are English); about 300+ answers per question are needed for stable discovery; models were validated on this one survey only.

## Update 3: themes for any new survey (default 10, fully editable)

Point a config at any file and the themes are generated from that data:

1. Discovery produces every topic the data supports (candidates). If fewer than `themes.n_themes` are found it retries with smaller
   minimum cluster sizes (`themes.expand`); a theme holding over `split_large_frac` of all clauses is split into sub-themes.
2. The `themes.n_themes` (default 10) largest are kept. Every smaller one is folded into its most similar kept theme, so its
   clauses are spread over the kept themes instead of being discarded (e.g. 50 found, 10 kept, the other 40 merged into them).
3. Your edits are applied: add, remove, merge, rename, or change the number of themes.

Interactive (`survey-nlp run --config <file>`): it asks for required themes and the count, shows the discovered themes with sizes and keywords, then
accepts commands until you press Enter: `n 8`, `add Price: cost, cheap`, `remove Cough, Cherry`, `merge Easy + Bottle = Opening`,
`rename Energy = Refreshing`, `reset`. Each edit re-shows the list. An edit that cannot apply is undone with an explanation. Choices are
saved in `<run_dir>/theme_edits.json` and offered again next time. Non-interactive: `--n-themes 8`, `--themes "Packaging,Price"`,
`--no-prompt` (reuses saved choices), or the `themes` section of the config.

Long-format files (one comment per row, with a question column and a text column) are supported with `input.question_col` and `input.text_col`;
see `config_open_ends.yaml`. Config edits that name a theme the data did not produce are skipped with a warning; edits typed at the prompt are strict.
Note: the Coco Pops themes changed slightly after these updates (an apostrophe repair altered the text), so Milk and Pieces are now defined by hand in `config.yaml`.

## Update 4: accuracy fixes, any-file mode and the Streamlit app

**Web app:** `pip install -r requirements.txt` then `streamlit run app.py`. Four steps: pick or upload a file (layout, header row and columns are
detected and can be corrected), find themes and edit them (rename, remove, merge, add, change how many), run, then coloured per-question tables, charts and
a download button.

**Any similar file from the command line:** `survey-nlp run --config config_default.yaml --input data/your_file.xlsx` (results in `runs/<file name>/`).
Detected automatically: header row (`input.header_row: auto`), layout (`input.layout: auto`: one comment per row with a question column, or one column
per question), open-ended columns, a question-ID row under the header, short sheet names ("Likes", "Dislikes") and each question's polarity from its name.

**What reading the beverage-survey results showed and what changed**

| Finding | Change |
|---|---|
| 36% of Like and 26% of Dislike answers had no theme, even "The overall taste is nice" | Clauses nothing else matched join their closest theme (`assign.fallback_min_sim`); the match threshold for meaning-only matching is calibrated (`assign.min_sim` 0.30). Answers with at least one theme: Like 64% to 88%, Dislike 74% to 87% |
| Keyword matches from discovered words were right only about 1 time in 3 (hand-checked, 37 pairs); matches by meaning about 95% | `assign.use_discovered_keywords: false` is the default (meaning only). Hand-written keywords still work. Overall hand-checked precision of the sample: 57% to 85% |
| The Coco-survey "flip weak negatives to positive" rule turned real complaints into praise here | `output.question_polarity` is now opt-in per question; the automatic rule (`output.auto_polarity`) only changes neutral aspect ratings and whole-answer ratings of bare mentions (4 words or fewer) |
| A clause could match 4 or 5 themes | `assign.max_themes_per_clause` (2 in the default config) keeps the strongest |

Hand labels in these checks were made by Claude, not a human. Remaining known errors: negation inside a theme ("Not too sweet" rated negative for sweetness), and
discovered theme names that do not describe their content well (for example a taste-in-general group named "Orange"); edit those in the app.

## Update 5: agreement with a human codebook (beverage survey, Eleni's reference)

**Why.** On `data/open_ends.xlsx` (500 answers, 250 Like / 250 Dislike) the pipeline run with `config_default.yaml` (`mode: spec`) disagreed with a
human coder: it gave 86 Neutral where she has 26, it discovered themes that split by sentiment ("Orange Flavour" = positive flavour, "Flavour" = negative
flavour) and it found none of Sweetness, Mouthfeel, Aftertaste, Aroma, Price, Brand. Her labels follow the question (92% of overall sentiments, 96% of theme
polarities are "Like = Positive, Dislike = Negative"), with real exceptions only where the answer itself says the opposite.

**What changed** (output format is unchanged; only values and theme names differ; survey-specific content is only in `config_beverage.yaml`):

| Area | Change |
|---|---|
| Evaluation | `eval/compare_to_reference.py` scores a workbook against a human sheet: overall accuracy / macro-F1 / confusion, per-theme presence P/R/F1, Positive/Negative agreement where both coded (never the "% same" cell: blank-vs-blank inflates it). `--split even|odd` scores half of the rows. `eval/check_format.py` asserts the format of two workbooks matches |
| Config | New `config_beverage.yaml`: `mode: extended`, Like/Dislike `question_polarity`, the 15 codebook themes as `themes.required` (regex keywords + seeds), `learn_keywords: false`, `required_merge_cos: 0.95`, `max_themes_per_clause: 2`, `min_sim 0.5`, `fallback_min_sim 0.3`, `curation.drop` for two leftover discovered themes that duplicated Flavour and Mouthfeel |
| Every answer gets a theme | `assign.py`: restored `fallback_min_sim` and `max_themes_per_clause` (the working copy no longer had them). A required theme with `general: true` (Non-specific) receives generic remarks and every clause nothing else matched (`assign.general_catch_all`, default on); the `assign.general_examples` exclusion does not fire when such a theme exists. "No theme" column kept (now empty) |
| Required themes | `semantic_only: true` (no keywords at all, e.g. Non-specific), `general: true`, `themes.learn_keywords: false` (only the keywords you wrote; words learned from the sample such as "good" or "bitter" put clauses in wrong themes). `curation.drop` may remove every discovered theme when required themes exist |
| Lukewarm rule | `output.neutral_phrases` (regex list in the config): an answer matching one, with no clearly evaluative word left once the phrase is removed and none of `output.neutral_blockers` ("too", "lacks"...), is Neutral in Overall and in every theme. The list was written from general English, not from the reference's Neutral texts |
| Opposite wording | `output.complaint_phrases` ("too sweet", "not ... enough", "lacks"; "not too sweet" is excluded): a Like answer containing one is Negative even if the model is unsure. Existing rules unchanged: evaluative opposite word, or model p >= 0.95 on a firm text |
| Negation in a theme | The existing `lexical_fix` already handles "not too sweet", "not so artificial", "wasn't as artificial as": in the even rows every such case came out Positive for Sweetness / Artificial, matching the reference. No change needed |
| CLI bug | `survey-nlp run --no-prompt` overwrote `themes.required` with an empty saved list, so the config's required themes were ignored. It now only overrides when the user chose some |

**Results** (reference = Eleni, same 500 rows). Tuning (keywords, thresholds, phrase lists) used the **even rows only**; the **odd rows were scored once at the end**.

| | before (all) | after: even (tuned on) | after: odd (held-out) | after (all) |
|---|---|---|---|---|
| Overall accuracy | 0.794 | 0.956 | **0.920** | 0.938 |
| Overall macro-F1 | 0.658 | 0.759 | **0.752** | 0.757 |
| Theme detection, weighted F1 | 0.474 | 0.865 | **0.855** | 0.860 |
| Polarity agreement (cells both coded) | 0.834 | 0.979 | **0.966** | 0.973 |
| Answers with no theme / mean themes per answer | 64 / 1.0 | | | 0 / 1.68 (reference 1.71) |
| Overall Neutral predicted (reference 26) | 86 | | | 8 |

Per-theme F1, before to after (even / odd / all): Flavour 0.74/0.78/0.76 to 0.91/0.92/0.92; Sweetness 0 to 0.89/0.87/0.88; Mouthfeel 0 to 0.87/0.85/0.86; Aftertaste 0 to 0.87/0.96/0.91;
Aroma 0 to 0.91/0.89/0.90; Artificial 0.86/0.79/0.83 to 0.89/0.94/0.91; Medicine 0.86/0.77/0.81 to 1.00/0.93/0.96; Packaging 0.25/0.28/0.26 to 0.74/0.83/0.78; Design 0.50/0.38/0.45 to 0.72/0.73/0.72;
Colour 0.40/0.29/0.34 to 0.88/0.88/0.88; Energy 0.57/0.64/0.61 to 0.86/0.84/0.85; Refreshing 0.46/0.30/0.39 to 0.91/0.83/0.87; Price 0 to 0.67/0.50/0.55; Brand 0 to 0.67/0.57/0.63; Non-specific 0 to 0.42/0.36/0.39.
The "before" run is `config_default.yaml` on the same file; its theme names were mapped to the codebook groups by `compare_to_reference.py`.

**Caveats.** The odd half is held-out for tuning, but it comes from the same file, the same survey and the same single coder, so it is not an independent sample; a second coder or a new survey would likely score lower.
Even-half tuning looked at wrong answers per theme (not at the reference's Neutral texts). Keywords and phrase lists are general English for each topic, not copied answers.
Hand labels from one coder (Eleni) are the only ground truth.

**Known weak spots.**
- Neutral: only 6 of 26 found (12 are called Negative, 8 Positive). Macro-F1 0.752 is just over the 0.75 target and depends on this small class. The rule catches lukewarm wording, not purely descriptive answers.
- Price: precision 0.40 (20 predicted, 9 in the reference); Non-specific F1 0.39 (generic comments compete with specific themes; the catch-all also receives odd leftovers); Brand recall 0.56.
- Packaging vs Design: "the packaging is attractive" is coded Design only by the reference, we give both.
- Artificial: "not artificial" answers are given Artificial = Positive; the reference has no Artificial (+) column, so those count as false positives for presence.
- Theme discovery is not stable when seeds change: the names of the 2 leftover discovered themes changed between runs, so `curation.drop` lists the names seen; any new leftover name would stay in the output as an extra column.
- With `question_col` set (long format, as in `config_open_ends.yaml`) the workbook has no side-by-side `results` / `results_codes` sheets (existing behaviour, unchanged); with `layout: auto` it does.
- The `themes` sheet lists different columns (`curated_keywords`, `seeded`, `required`, `general` instead of `assign_keywords`, `candidates_found`) because it is a dump of the theme records.

**Streamlit app.** The app used to build its settings from `config_default.yaml` only (`mode: spec`, so the Like / Dislike rules were off). Data tab > Settings now has a **Settings preset** selector listing every `config*.yaml` next to `app.py` (default `config_default.yaml`, generic): its fixed themes, keywords, rating rules and thresholds are used, the question selectors start from the preset's polarity, and the table can remove or rename a fixed theme. Pick `config_beverage.yaml` only for this survey's codebook. A run through the app on `open_ends.xlsx` gives the same numbers as the command line.

**Number of themes.** In the app the slider is now the TOTAL number of themes (fixed ones included); a "No theme" column is always added on top. It sets `themes.n_total`: when there are more themes than that, the smallest are dropped, themes found in the answers first, then fixed ones (asking for 10 with the 15-theme beverage preset gives its 10 largest). The config key can be used from the command line too.

**Generic by default.** `config_default.yaml` is now `mode: extended` and carries the general-English lists (`complaint_phrases`, `neutral_phrases`, `neutral_blockers`), so any dataset gets the rating rules with no codebook; the polarity of each question is read from its answers (`auto_polarity`) unless set in the app. Only the theme names and topic keywords are survey-specific, and they live in an optional preset such as `config_beverage.yaml`. On the beverage file with the generic default: overall accuracy 0.938, macro-F1 0.757, polarity agreement 0.970 (the same rating rules), theme F1 0.48 because themes are found from the answers instead of the codebook, 20 of 500 answers without a theme.

**Automatic in the app (supersedes the preset selector above).** The app has no settings to choose: it always uses the built-in `config_default.yaml`. Themes are discovered from the answers, each question's polarity is read from its answers (it can be overridden per question), the general-English rating rules are on, and a built-in "General comment" theme (matched by meaning, not by keywords) receives generic remarks and every clause nothing else matched, so every answer has a theme. Other `config*.yaml` files are still used from the command line (`survey-nlp run --config ...`). On the beverage file this way: overall accuracy 0.936, macro-F1 0.755, polarity agreement 0.966, 0 of 500 answers without a theme. Agreement of the discovered theme names with Eleni's codebook is low (weighted F1 0.36), as the names are discovered, not given.
