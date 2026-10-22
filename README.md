# Survey NLP: open-ended answers to themes and per-theme sentiment

Reads the open-ended answers of a survey (Excel or CSV), finds the topics people talk about (themes), and rates how people feel about each
theme (Positive / Negative / Neutral) for every answer. Results come as a coloured Excel workbook, with a web app and a command line to run it.

Everything runs locally on CPU or GPU with open models: no data is sent to an outside service. English answers only.

## What it does

1. **Reads the file.** Excel or CSV. The header row, the layout (one comment per row with a question column, or one column per question) and the
   open-ended columns are detected automatically and can be corrected. Typos are repaired in the text the models read; the answer shown in the
   results is unchanged.
2. **Splits answers into clauses.** "Tastes great but too sweet" becomes two clauses, so each part can have its own theme and rating.
3. **Finds themes from the answers.** The number you ask for is the number you get. Themes can be renamed, merged, removed or added by hand, and
   you can list themes that must be in the results.
4. **Assigns clauses to themes** mainly by meaning (embedding similarity), plus keywords you wrote. A built-in "General comment" theme takes generic
   remarks ("it is good") and anything no other theme matched, so every answer has a theme.
5. **Rates sentiment** for the whole answer (Overall) and for every theme in it (aspect-based sentiment).
6. **Exports** one workbook with a sheet per question, a combined sheet, the numeric codes, the detail behind each rating, and the themes.

## Run it

Install: `pip install -r requirements.txt` (the package is also installable with `pip install -e .`, which adds the `survey-nlp` command).

**Web app:** `streamlit run app.py`. Four steps:

| Step | What you do |
|---|---|
| Data | Pick or upload a file; check the detected layout, header row and columns; type how many themes you want |
| Themes | See the themes found with their main words and example answers; rename, merge, remove or add themes |
| Analyse | Run the sentiment models for every theme |
| Results | Coloured tables per question, charts, and a download button |

The app has no settings to choose: it always uses the built-in `config_default.yaml`.

**Command line:**

```
survey-nlp run --config config_default.yaml --input data/your_file.xlsx
```

Results go to `runs/<file name>/aspect_sentiment_results.xlsx`. Useful options: `--n-themes 8`, `--themes "Packaging,Price"` (themes that must be
included), `--no-prompt` (reuse saved choices), `--until themes` (stop after theme discovery), `--force` (ignore saved results), `--rediscover`
(find the clusters again). Without `--no-prompt` it shows the discovered themes and accepts edits: `n 8`, `add Price: cost, cheap`,
`remove Cough, Cherry`, `merge Easy + Bottle = Opening`, `rename Energy = Refreshing`, `reset`.

## How the pipeline works

The pipeline runs in stages, and each stage saves its result in the run folder. A stage is only redone when its input, its settings or its code
changed, so a re-run with nothing changed is instant and a change to a late setting does not redo the early stages.

| Stage | File | What it does |
|---|---|---|
| ingest | `ingest.py`, `spelling.py` | Reads the file, detects header row, layout and open-ended columns, drops question-ID rows and "nothing to say" answers (by pattern and by meaning), repairs typos |
| segment | `segment.py` | Splits each answer into clauses at contrast words ("but", "however"...) and sentence ends |
| embed | `embed.py`, `cache.py` | Turns each clause into a vector with `BAAI/bge-small-en-v1.5`; vectors are cached on disk per text, and the model is loaded once per run |
| themes | `themes.py`, `quality.py` | Finds, names and checks the themes (see below) |
| assign | `assign.py` | Gives each clause its themes: by meaning, by keyword, by fallback, or General comment |
| overall | `sentiment.py` | Rates each whole answer with `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| absa | `sentiment.py` | Rates each (clause, theme) pair with `yangheng/deberta-v3-base-absa-v1.1` |
| export | `export.py` | Combines the ratings and writes the workbook |

### How themes are found

- **Candidates from words.** The topic words of the answers (nouns and plain attributes, without judging or filler words) are grouped by what
  the clauses using them are about. Each clause goes to the group most of its words belong to. There is no random step, so the same data gives the
  same themes. A group holding too large a share of the answers is split.
- **Small surveys.** If fewer themes than asked for are found, looser attempts follow: a smaller minimum theme size scaled to the data, rarer words
  as topic words, then dense regions of clauses (UMAP + HDBSCAN).
- **Exact count.** If there are still too few, the largest theme is split in two until the number asked for is reached. If there are too many, the
  smallest are folded into their most similar kept theme, so their clauses are not lost.
- **Cleaning.** Near-duplicate themes are merged; oversized ones are split; generic clusters ("easy drink, overall") are folded into General comment.
- **Names and main words.** A theme is named after the noun or plain attribute that best summarises its clauses, and the name is checked against
  the clauses (a name that few of them use is replaced, or the theme is split). Main words are noun chunks and short phrases ranked by closeness in
  meaning to the theme and how much more the theme uses them than the whole survey.
- **Judging words** ("nice", "unpleasant") are not topics. They are never used as names or main words.
- **Your themes.** Required themes are matched by their name, keywords and example phrases and take the clauses closest to them.
- **Stable results.** The clusters are saved in `clusters.json`, keyed by the clauses, the settings and the clustering code, so adding a theme
  or re-running does not reshuffle them.

### How sentiment is decided

Each rating is Positive, Negative or Neutral. The model's probabilities are turned into a rating with an evidence threshold and an allowance
(`absa.evidence_threshold`, `absa.allowance`). On top of that, in `mode: extended`: negation is handled ("not too sweet"), a Like question's bare
mentions count as Positive and a Dislike question's as Negative (the polarity of each question is read from its answers unless set), lukewarm
wording is Neutral, and complaint wording in a Like answer is Negative. `mode: spec` uses the model's ratings only.

## The output workbook

`aspect_sentiment_results.xlsx` has:

| Sheet | Contents |
|---|---|
| One per question | Each answer, its Overall rating and one column per theme, coloured by rating |
| Combined | Every question's answers one under the other |
| results / results_codes | Side by side: words (Positive / Negative / Neutral) and codes (0 / 1 / 2); only when the layout has one column per question |
| detail | The raw model result and the reason behind each rating |
| pairs | Every (clause, theme) pair with its similarity and how it matched |
| themes | The theme records: name, main words, size, example clauses, name quality |
| spelling | The typos that were repaired |

## Files

```
app.py                 Streamlit web app (the 4 steps above)
config_default.yaml    Built-in settings: works on any survey with no editing (used by the app)
config.yaml            Settings tuned for the Coco Pops tasting survey (data/coco_pops_VR.xlsx)
config_beverage.yaml   Settings with the beverage codebook's themes (data/open_ends.xlsx)
requirements.txt       Python packages
pyproject.toml         Package definition; adds the `survey-nlp` command
Dockerfile             Container for Hugging Face Spaces (port 8501)
.devcontainer/         Dev container settings
data/                  Example survey files (not committed)
scripts/
  download_models.py   Downloads the models ahead of time (used by the Dockerfile)
eval/                  Quality and accuracy checks
  quality_eval.py      Quality report of the themes of a run
  run_suite.py         Runs the pipeline on the datasets in eval/suite and compares with a saved baseline
  compare_models.py    Compares embedding / sentiment models
  suite/               Small synthetic datasets used by the suite
tests/                 Unit tests (pytest), one file per module
PROJECT_SPEC.md        The original specification the project was built from
PROJECT_OVERVIEW.md    Design notes and the reasons behind the choices
PSEUDOCODE.md          The algorithms in plain steps
CHANGELOG.md           What changed, version by version
src/survey_nlp/
  cli.py               The `survey-nlp` command and the interactive theme editor
  pipeline.py          Runs the stages in order with saved results
  config.py            Loads a config file
  ingest.py            Reads and cleans the file, detects layout and columns
  spelling.py          Typo repair that protects names and survey words
  segment.py           Splits answers into clauses
  embed.py, cache.py   Embedding model and the on-disk vector cache
  themes.py            Theme discovery, naming, merging, splitting, your own themes
  assign.py            Matches clauses to themes
  sentiment.py         Overall and per-theme sentiment, and the rating rules
  quality.py           Checks of theme quality (name coverage, duplicates, sizes)
  export.py            Writes the workbook
  util.py              Shared helpers (hashing, device choice, centring embeddings)
```

## Settings

All settings are in a YAML config; the main sections are:

| Section | Controls |
|---|---|
| `input` | File, sheet, header row, layout, text columns, non-answer patterns, spelling repair |
| `segment` | Words that split an answer into clauses |
| `embed` | Embedding model, batch size, cache folder |
| `themes` | Number of themes, discovery thresholds, required themes, hand edits (`curation`), naming and keyword rules |
| `assign` | How close a clause must be to a theme (`min_sim`), keyword and fallback rules, themes per clause |
| `overall`, `absa` | Sentiment models and the evidence threshold and allowance |
| `output` | Labels, question names, polarity of each question, the neutral and complaint wording lists |

`themes.curation` edits (merge, rename, drop, extra keywords, seeds, add) use theme names from a particular survey, so they are written per survey.
A name the data did not produce is skipped with a warning.

## Tests

`python -m pytest tests -q`

## Limits

- English only: the models are English.
- Needs enough answers for themes to be stable. Small files still give the number of themes you ask for, but the smallest ones are less distinct.
- Sentiment models are weaker on negation and sarcasm inside a theme.
- Theme names are chosen by rules and are sometimes poor; rename them in the app.
- Cached embeddings and run folders live on disk (`cache/`, `runs/`, both git-ignored). On a hosted app a restart can clear them, so the next run is slower.
