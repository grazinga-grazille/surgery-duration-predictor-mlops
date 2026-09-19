"""
Central configuration for the surgical stopword / TF-IDF pipeline.

Step 0 — All file paths, column names, thresholds, and the random seed live here.
Every other module imports from this file so magic numbers are never scattered.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project roots
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Input data
# ---------------------------------------------------------------------------
# Prefer the copy under data/raw/; fall back is documented in run_pipeline.
INPUT_FILE = DATA_RAW / "LoadFile.csv"

# Free-text procedure description only — never use CPT / mnemonic / structured
# procedure-type fields as procedure identity for this pipeline.
TEXT_COLUMN = "ProcedureDescription"
DURATION_COLUMN = "ActualDurationMinutes"
CASE_ID_COLUMN = "case_id"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Step 1 — Preprocessing
# ---------------------------------------------------------------------------
# Minimum duration (minutes) retained after null / non-positive filtering.
MIN_DURATION_MINUTES = 0.0  # exclusive: duration must be > this value

# Truncation length for example snippets in candidate_stopwords.csv
EXAMPLE_SNIPPET_CHARS = 80

# Baseline stopwords applied in Step 1 (before phrase detection), always removed.
# NLTK English stopwords are loaded at runtime and unioned with this list.
# Forms are lowercase post-normalization tokens (W/WO -> "wo" after slash expand).
EXTRA_BASELINE_STOPWORDS = frozenset(
    {
        "with",
        "wo",  # from W/WO / w/wo after slash normalization
        "w/wo",  # defensive, if seen before slash expand
        "and",
        "of",
        "s",  # leftover from "(S)" side-marker if parentheses strip first
    }
)
# ---------------------------------------------------------------------------
# Step 2 — Phrase detection (gensim Phrases / Phraser)
# Mikolov et al. (2013) scoring; two-pass approach matching Sarica & Luo (2021).
# Defaults mirror gensim; paper used (5, 2.5) on patents — tune after inspection.
# ---------------------------------------------------------------------------
PHRASE_MIN_COUNT = 5
PHRASE_THRESHOLD_PASS1 = 10.0  # stricter: frequent, obvious bigrams
PHRASE_THRESHOLD_PASS2 = 5.0   # looser: trigrams / four-grams from bigrams
PHRASE_DELIMITER = "_"         # gensim default connector between phrase parts

# ---------------------------------------------------------------------------
# Step 3 — Stopword candidate discovery (Sarica & Luo 2021 metrics)
# ---------------------------------------------------------------------------
# Top-N per ranked metric list. Paper used 2000 on a huge patent vocab;
# surgical corpora are much smaller — 500 is a sensible starting default.
STOPWORD_TOP_N = 500

# ---------------------------------------------------------------------------
# Step 5 — Duration cross-check (point-biserial + Mann-Whitney)
# A human-"remove" token is blocked from the final stopword list if it shows
# significant association with ActualDurationMinutes.
# ---------------------------------------------------------------------------
# Absolute point-biserial |r| above this -> treat as having duration signal.
DURATION_PBISER_ABS_THRESHOLD = 0.05
# Mann-Whitney two-sided p-value below this -> treat as having duration signal.
DURATION_MW_P_THRESHOLD = 0.05
# Minimum cases with / without the token for a valid statistical test.
DURATION_MIN_GROUP_SIZE = 30

# ---------------------------------------------------------------------------
# Intermediate / output artifact paths
# ---------------------------------------------------------------------------
TOKENIZED_CORPUS_PATH = DATA_INTERIM / "tokenized_corpus.pkl"
CASES_CLEANED_PATH = DATA_INTERIM / "cases_cleaned.parquet"

PHRASE_MODEL_PASS1_PATH = DATA_INTERIM / "phrases_pass1.pkl"
PHRASE_MODEL_PASS2_PATH = DATA_INTERIM / "phrases_pass2.pkl"
PHRASER_PATH = DATA_INTERIM / "phraser.pkl"
PHRASE_CORPUS_PATH = DATA_INTERIM / "phrase_corpus.pkl"
PHRASE_FREQ_PATH = DATA_INTERIM / "phrase_frequencies.csv"

CANDIDATE_STOPWORDS_PATH = OUTPUTS / "candidate_stopwords.csv"
DURATION_CROSSCHECK_PATH = OUTPUTS / "duration_crosscheck.csv"
FINAL_STOPWORDS_PATH = OUTPUTS / "final_stopwords.txt"

STOPWORD_FILTERED_CORPUS_PATH = DATA_PROCESSED / "stopword_filtered_corpus.pkl"
CASES_FINAL_PATH = DATA_PROCESSED / "cases_final.parquet"
TFIDF_MATRIX_PATH = DATA_PROCESSED / "tfidf_matrix.npz"
TFIDF_VECTORIZER_PATH = DATA_PROCESSED / "tfidf_vectorizer.joblib"
TFIDF_FEATURE_NAMES_PATH = DATA_PROCESSED / "tfidf_feature_names.txt"
# Row i of the TF-IDF matrix maps to this table's row i (case_id + duration).
TFIDF_ROW_INDEX_PATH = DATA_PROCESSED / "tfidf_row_index.parquet"

# ---------------------------------------------------------------------------
# Step 7 — Final TF-IDF embeddings (sklearn; standard formula OK here)
# Input: phrase-joined cleaned_text from preprocessing (interim by default).
# The Sarica & Luo harsher TF-IDF variant is ONLY for Step 3 discovery.
# ---------------------------------------------------------------------------
# Source table with cleaned_text + case_id + duration (post Step 1-2).
# Defaults to interim cases_cleaned.parquet — your finalized preprocess output.
# Point at CASES_FINAL_PATH instead if you later apply vetted domain stopwords.
TFIDF_INPUT_CASES_PATH = CASES_CLEANED_PATH

# Cap vocabulary size. Terms are ranked by corpus term frequency; only the
# top ``max_features`` are kept. Surgical procedure text has a small vocab
# (~hundreds-low thousands), so 2000-5000 is a soft cap that still leaves
# room to grow. None = no hard cap (still filtered by min_df / max_df).
TFIDF_MAX_FEATURES = 5000

# Drop terms that appear in fewer than this many *documents* (cases).
# Raises reliability: rare one-off spellings / typos won't become features.
# Typical range 2-5; 2 is permissive, 5 is stricter.
TFIDF_MIN_DF = 2

# Drop terms that appear in more than this fraction of documents.
# Catches near-universal leftovers even after baseline stopword removal
# (e.g. a word in >=90% of cases). Float in (0, 1]; 0.9 is a common default.
TFIDF_MAX_DF = 0.9

# N-gram range passed to TfidfVectorizer.
# Default (1, 1) = unigrams only. Multi-word surgical phrases were already
# glued into single underscore tokens in Step 2 (gensim Phrases), e.g.
# ``cesarean_section``, so we do NOT also ask sklearn to emit its own
# bigrams/trigrams — that would double-count and inflate sparsity.
TFIDF_NGRAM_RANGE = (1, 1)
