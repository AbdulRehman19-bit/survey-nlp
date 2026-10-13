# Survey NLP: pseudo code

Written for: developers and reviewers who want the logic of the application without reading the source.
It follows the code in `src/survey_nlp/` (modules named in brackets) and `app.py`. Numbers in brackets are the defaults from `config_default.yaml`.

---

## 1. Top level

```
PROGRAM survey_nlp
    INPUT   survey file (xlsx/csv), config (yaml), optional overrides from the user
    OUTPUT  Excel workbook: one coloured sheet per question + detail, themes, quality, spelling sheets

    cfg <- load config, apply overrides, set random seed
    art <- run_pipeline(cfg)            # section 2
    write workbook from art             # section 9
```

Two front ends call the same `pipeline.run`:

```
CLI (cli.py):   ask for required themes and number of themes
                run pipeline until "themes"
                show discovered themes, let user edit them (n / add / remove / merge / rename / reset)
                run pipeline until "export"

WEB APP (app.py, Streamlit):
                upload file -> preview sheet and header row
                choose preset or settings -> build_config
                "Find themes" button -> run_stage(until = "themes")
                show themes table, let user edit -> overrides_for(cfg, edits)
                "Run" button -> run_stage(until = "export")
                show sentiment bars per theme, download workbook
```

---

## 2. Pipeline with caching [pipeline.py]

```
PLAN = ordered list of (stage, parent stages, config sections that invalidate it)
    ingest  <- []                 input, embed.model
    segment <- ingest             segment
    embed   <- segment            embed
    themes  <- embed              themes, seed, assign.min_sim
    assign  <- themes             assign
    overall <- ingest             overall
    absa    <- assign             absa model/batch/length/quantize
    export  <- absa, overall      output

FUNCTION run(config, until, force, overrides)
    cfg <- load(config); merge overrides; set_seed
    FOR EACH (name, parents, sections) IN PLAN
        key <- hash( input-file signature (ingest only) OR parents' keys,
                     source code of the functions this stage runs,
                     hash of the config sections listed for the stage )
        IF name != export AND not force AND saved result exists AND saved key == key
            art[name] <- load saved result              # "cached"
        ELSE
            art[name] <- compute(name)
            IF name != export: save result and key
        IF name == until: STOP
    RETURN art
```

Changing a setting or editing the code therefore re-runs only the affected stage and the stages after it.

```
FUNCTION scopes(cfg, long, clauses)
    IF themes.scope == "pooled": RETURN { ALL: every clause }
    ELSE RETURN { question: clauses of that question's answers }  for each question
```

Themes and assignment are done once per scope; sentiment is scored over all of them.

---

## 3. Stage 1: ingest [ingest.py, spelling.py]

```
FUNCTION ingest.run(cfg)
    df <- read_table(path, sheet, header row)       # header row guessed when "auto"
    IF layout is auto AND no explicit columns
        layout <- detect_layout(df)                 # "long": a question column + a text column
                                                    # "wide": one open-ended column per question
    IF long layout: check question_col and text_col exist
    ELSE text_cols <- given, or detect_text_cols(df)   # columns with mean words >= 4 and unique ratio >= 0.3
         if none found: ERROR
    drop a leading row that only holds ids
    build `long` table: one row per (respondent, question, text)

    clean <- replace U+FFFD with apostrophe, squeeze whitespace, trim
    IF spell_correct
        find words written fewer than spell_min_count times that are close to a frequent word
        (skip proper nouns and protected words), replace them in the text the models read
        record every change for the "spelling" sheet
    valid <- length(clean) >= min_chars
    valid <- valid AND NOT matches non_answer_regex            # "n/a", "nothing", "no complaints" ...
    valid <- valid AND NOT (embedding similarity to any non_answer_example >= 0.8)
    text_key <- hash(lowercase clean text) for valid rows
    RETURN long, base                                 # base = one row per respondent (id, kept columns)
```

---

## 4. Stage 2: segment into clauses [segment.py]

```
FUNCTION segment.run(unique valid texts, cfg)
    pattern <- optional comma/semicolon + one of (but, however, although, though, yet, whereas)
    FOR EACH unique text
        split into sentences (rule-based sentencizer)
        FOR EACH sentence
            split at the contrast pattern
            FOR EACH piece: strip punctuation
                IF piece has >= min_words (2) words: keep as a clause
        IF no clause kept: keep the whole text as one clause
    give every clause a clause_id and a hash clause_key
```

"Smell bad but taste ok" -> clauses "Smell bad", "taste ok".

---

## 5. Stage 3: embed [embed.py, cache.py]

```
FUNCTION embed.run(clauses, cfg)
    FOR EACH clause: look up its vector in the on-disk cache by clause_key
    encode the missing ones with BAAI/bge-small-en-v1.5 (384 dims, normalised, batch 128)
    store new vectors in the cache
    RETURN matrix E (n_clauses x 384)
```

The same cached encoder is used for seed phrases, non-answer examples and general examples.

---

## 6. Stage 4: theme discovery [themes.py]

```
FUNCTION discover(clauses, E, cfg)  ->  (themes, C)       # C = one unit vector per theme

    # 6.1 prepare
    ok   <- clauses with >= min_words words (sample max_fit_clauses if more)
    ERROR if fewer than 2 * min_cluster_size usable clauses
    mu   <- mean of E
    axis <- positive-minus-negative direction built from sentiment_axis example phrases
    Ec   <- E with mu and 70 % of the axis removed         # "centred" space: topics, not tone
    Ecl  <- E with the axis fully removed                  # clustering space
    gv   <- centre of the built-in general theme ("Not Specific") from its seed phrases
    pool <- ok without clauses whose similarity to gv >= 0.4   # generic remarks do not form topics
    noise, generic <- words that never name a theme (general English filler / judging words)

    # 6.2 candidates (frozen on disk, keyed by a fingerprint of the data + config)
    IF a frozen result exists for this fingerprint: load it
    ELSE
        add aspect terms (noun phrases) to Ecl
        merged <- aspect_candidates(pool)        # group clauses by frequent aspect nouns, rank by lift
        IF too few: merged <- density clusters   # UMAP (5 dims) + HDBSCAN (min size 6)
        save

    # 6.3 clean up the candidate set
    merged <- merge_duplicates(merged)           # centre cosine > 0.92, or same name/keywords (MMR check)
    merged <- split_oversized(merged)            # a theme > 30 % of clauses is split if it separates
    merged <- reduce(merged, n_themes)           # keep the largest n_themes
    ERROR if fewer than min_themes
    themes <- name_and_split(merged)
    FOR EACH theme
        candidate names <- head nouns/phrases of its clauses (no filler words), best first
        pick the first name that
            is used by >= name_min_coverage (25 %) of the theme's clauses, and
            is a noun not already used by another theme
        IF none fits: call the theme "mixed" and split it if both halves can be named
        keywords <- n_phrases (6) noun chunks chosen with MMR (relevance vs variety)
    fold_generic(themes)                         # themes that are really "it is good" join the general theme
    recheck_names(themes)                        # name must still cover its final members
    themes, merged <- curate(themes)             # user ops: remove, merge, rename

    # 6.4 centres
    C[j] <- unit(mean(Ec[members of theme j]))
    FOR EACH curation.seeds: C[j] <- unit(C[j] + unit(mean(seed vectors)))      # sharpen a theme
    FOR EACH curation.add (user-defined theme)
        centre <- mean of seed vectors, expanded with the closest clauses (expand)
        learn keywords close in meaning to the theme (>= 0.35)
        IF a theme with that name exists: sharpen it ELSE append as new theme
    apply_required(themes)                       # themes that must exist, incl. the general one
    REPEAT 2 times (refine)
        FOR EACH theme: C[j] <- unit(C[j] + unit(mean of clauses now closest to it and >= assign.min_sim))
    IF n_total: trim to that many themes

    # 6.5 statistics
    FOR EACH theme
        frequency      <- number of clauses nearest to it with similarity >= floor
        theme.min_sim  <- 5th percentile of its members' similarity, clamped (tight themes are stricter)
        representative_clauses <- n_examples (3) clauses nearest the centre
    RETURN themes, C
```

---

## 7. Stage 5: assign clauses to themes [assign.py]

```
FUNCTION assign.run(clauses, E, themes, C, cfg)
    E_c <- centre E the same way as in discovery
    S   <- E_c x C^T                                   # similarity of every clause to every theme

    # semantic match
    best, second <- the two highest-scoring themes per clause
    A[best]   <- best_sim >= min_sim (0.4)
    A[second] <- second_sim >= min_sim AND (best_sim - second_sim) <= secondary_margin (0.1)

    # keyword match
    FOR EACH theme
        words <- user-written keywords, ELSE distinctive words of the theme's own name
        K[:, theme] <- clause contains a word (whole word, case-insensitive; "re:" prefix = regex)
        IF the words were not written by the user:  K <- K AND (S >= keyword_min_sim 0.25)
        user-written keywords are "trusted": they match whatever the similarity

    # clauses that are only generic remarks
    IF there is no general theme AND clause is >= 0.9 similar to a general example ("It is good"):
        remove its semantic matches

    # fallback for clauses nothing matched
    IF fallback_min_sim set:
        lone <- clause matched no theme
        lone clause joins its best SPECIFIC theme if similarity >= 0.25
              (and that is at least as close as the general theme)
    IF a general theme exists:
        clauses that have a specific theme lose the general one
        clauses still unmatched go to the general theme    # every answer ends with a theme

    # cap
    keep each clause's strongest max_themes_per_clause (2) themes
        strength = sim + 0.15 * semantic + 0.15 * keyword  (general theme: -0.5)
        a trusted keyword match is never dropped by the cap
    at answer level: a general match is dropped if another clause of the same answer has a specific theme

    via <- "both" | "semantic" | "keyword" | "fallback"
    RETURN table (clause_id, theme_idx, sim, via)
```

---

## 8. Stages 6 and 7: sentiment [sentiment.py]

```
FUNCTION run_overall(unique valid texts, cfg)           # one rating per answer
    model <- cardiffnlp/twitter-roberta-base-sentiment-latest
    P <- softmax probabilities (neg, neu, pos) in batches of 64, max length 256
    label <- argmax; confidence <- max
    RETURN (text_key, p_neg, p_neu, p_pos, label, confidence)

FUNCTION run_absa(clauses, assignments, themes, cfg)    # one rating per (clause, theme)
    pairs <- assignments joined with their clause text and theme name
    unique <- distinct (clause_key, scope, theme)       # score each unique pair once
    model <- yangheng/deberta-v3-base-absa-v1.1
    P <- probabilities for the sentence pair (clause, theme name)
    RETURN pairs with p_neg, p_neu, p_pos
```

Both models run offline, on GPU with fp16 if available, otherwise CPU.

---

## 9. Stage 8: export and final rating [export.py, sentiment.py, quality.py]

```
FUNCTION decide(p_pos, p_neg, threshold 0.6, allowance 0.05)
    pos <- p_pos >= threshold AND p_neg <= p_pos + allowance
    neg <- p_neg >= threshold AND p_pos <= p_neg + allowance
    neutral <- neither              # both may fire: a mixed opinion

FUNCTION aggregate(scored, cfg, valence)               # one code per (answer, scope, theme)
    p_pos, p_neg <- MAX over the answer's clauses for that theme
    code <- Positive if only pos fired, Negative if only neg fired, Neutral if neither
    IF both fired: mixed_policy "stronger" -> the larger probability wins; flag mixed
    IF mode is "extended" (not "spec")
        firm_neg   <- clause with >= 4 words that the model calls Negative at >= 0.95
        valence    <- word_valence(): words that the model rates consistently (min support 2, conf 0.65)
        lexical flags: clearly positive / negative words in the clause, with negation handling
                       ("not", "no", "less", "lack" ... reach up to 3 words, stop at "but", "and")
        lexical_fix: correct a code that contradicts the clear words ("not so artificial" is not Negative)
        complaint_phrases: "too sweet", "not enough", "lacks" ... count as a complaint
    RETURN table

FUNCTION export.run(art, cfg, timings)
    g <- aggregate(absa, cfg, valence)
    L <- long answers joined with overall ratings; overall_code <- code of the label
    IF extended mode: apply lexical_fix to the overall code too

    FOR EACH question
        name <- short sheet name
        polarity <- user setting, OR infer_polarity()   # is the question "what did you like" or "dislike"?
                    (>= 60 % of rated answers lean one way, ratio >= 3)
        IF polarity found
            a Neutral rating (whole answer, theme ratings) becomes the question's polarity,
                because merely mentioning a theme in a "like" question is praise
            a weak rating AGAINST the polarity is reset to it (model tripping over negation)
            unless the answer uses a clearly evaluative opposite word ("smell bad" in a "like" question)
            complaint phrases / lukewarm wording (neutral_phrases without neutral_blockers) override
        table <- one row per respondent: answer, overall rating, one column per theme
                 (code, or not_discussed_value when the theme is not mentioned)
        colour cells Positive / Negative / Neutral

    detail rows <- every (answer, clause, theme, via, similarity, probabilities, flags)
    write workbook:
        one sheet per question
        detail sheet, themes sheet (names, keywords, examples, size)
        quality sheet, spelling sheet, timings
    quality.compute(art, cfg): per theme
        name coverage and lift, coherence (NPMI of keywords), size spread (Gini),
        duplicates (keyword Jaccard), flags -> summary line + JSON
```

---

## 10. Web app flow [app.py]

```
ON PAGE LOAD
    sidebar: file upload, sheet, header row, preset (config_*.yaml), number of themes,
             required themes (text -> parse_required), mode, polarity per question
    s <- session state

build_config(s)  -> full config dict from the sidebar
overrides_for(cfg, edits) -> theme edits as config overrides
settings_sig(cfg, edits)  -> hash; changed hash means previous results are stale

ON "Find themes"
    run_stage(cfg, edits, until = "themes")     # shows a progress slot, quiet logs
    themes_table(art) -> name, keywords, size, examples
    user edits: rename, merge, remove, add (with keywords)

ON "Run"
    run_stage(cfg, edits, until = "export")
    results <- latest_results(run_dir)
    per question: sentiment_bar(pos, neu, neg) for each theme, coloured table (colour_cell)
    download button for the workbook (quality sheet hidden in the view)
```

---

## 11. Evaluation tools [eval/]

```
run_suite.py          run the pipeline over five datasets + a held-out set, compare with baseline_suite.json
quality_eval.py       theme quality report vs baseline_quality.json
theme_stability.py    re-run with different seeds, measure how stable the themes are
compare_models.py     compare embedding / sentiment model choices
evaluate.py, compare_to_reference.py, make_label_sheet.py, make_review_sheet.py
                      score ratings against hand-labelled reference sheets
tests/                unit tests per module + a suite test
```
