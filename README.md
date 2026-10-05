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
