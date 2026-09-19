"""
Step 1 — Preprocessing: clean, tokenize, POS-tag, and lemmatize procedure text.

NLP stack choice
----------------
We use **NLTK** (punkt + averaged_perceptron_tagger + WordNetLemmatizer) rather
than spaCy because:

1. It matches the Sarica & Luo (2021) description of POS-aware lemmatization
   without requiring a separate large language model download.
2. Dependencies stay lighter and easier to pin for research reproducibility.
3. POS -> WordNet tag mapping makes lemmatization reliable (unlike bare
   WordNetLemmatizer calls without POS).

Do **not** use CPT / mnemonic / structured procedure-type fields here — only
the free-text procedure description column.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Sequence, Set, Tuple

import nltk
import pandas as pd
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

from . import config
from .utils import ensure_dirs, load_table, save_pickle, setup_logging

logger = logging.getLogger("surgical_stopwords.preprocess")

# Punctuation at token boundaries / fully non-alphanumeric tokens.
# Hyphens and glued ampersands inside words (post-op, t&a, a&p) are preserved.
_BOUNDARY_PUNCT_RE = re.compile(r"^[^A-Za-z0-9&]+|[^A-Za-z0-9&]+$")
_ONLY_NON_ALNUM_RE = re.compile(r"^[^A-Za-z0-9&]+$")
# Pure numbers / punctuation with no alphabetic content.
_NO_ALPHA_RE = re.compile(r"^[^A-Za-z]+$")
# Clinically meaningful code-like tokens: letters+digits and/or internal - /
# e.g. l4-l5, t1-t2, covid-19 (slashes are normalized away before tokenize)
_CLINICAL_CODE_RE = re.compile(
    r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9]+(?:[-][A-Za-z0-9]+)*$"
    r"|^(?=.*\d)[A-Za-z]*\d+[A-Za-z]*(?:[-][A-Za-z0-9]+)+$",
    re.IGNORECASE,
)

# Placeholder so NLTK does not split glued compounds like "T&A" / "A&P".
# Standalone "&" (whitespace on either side) is left alone and may separate.
_AMP_PLACEHOLDER = "xxampxx"
_AMP_BETWEEN_ALNUM_RE = re.compile(r"(?<=[a-z0-9])&(?=[a-z0-9])")
_MULTI_SPACE_RE = re.compile(r"\s+")
# Laterality / laterality-style side marker "(S)" -> drop before tokenize
_SIDE_MARKER_RE = re.compile(r"\(\s*s\s*\)", re.IGNORECASE)

_LEMMATIZER = WordNetLemmatizer()
_BASELINE_STOPWORDS: Optional[Set[str]] = None


def ensure_nltk_resources() -> None:
    """
    Download required NLTK resources if missing (Step 1 prerequisite).

    Safe to call repeatedly; nltk.download is a no-op when data is present.
    """
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", name)
            nltk.download(name, quiet=True)


def get_baseline_stopwords() -> Set[str]:
    """
    NLTK English stopwords union domain extras from config (Step 1).

    Applied after lemmatization and before phrase detection so function words
    never enter phrase scoring or the Sarica & Luo candidate lists.
    """
    global _BASELINE_STOPWORDS
    if _BASELINE_STOPWORDS is None:
        ensure_nltk_resources()
        base = {w.lower() for w in nltk_stopwords.words("english")}
        extra = {w.lower() for w in config.EXTRA_BASELINE_STOPWORDS}
        _BASELINE_STOPWORDS = base | extra
        logger.info(
            "Baseline stopwords: nltk_english=%d extras=%d union=%d",
            len(base),
            len(extra),
            len(_BASELINE_STOPWORDS),
        )
    return _BASELINE_STOPWORDS


def _penn_to_wordnet(tag: str) -> str:
    """
    Map Penn Treebank POS tags to WordNet POS for accurate lemmatization.

    Step 1: bare WordNetLemmatizer without POS defaults to NOUN and is unreliable
    for verbs/adjectives (paper also used a POS tagger before lemmatizing).
    """
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def strip_boundary_punctuation(token: str) -> str:
    """
    Strip punctuation only at word boundaries (Step 1).

    Keeps internal hyphens/slashes used in medical compounds (post-op, x-ray).
    """
    if _ONLY_NON_ALNUM_RE.match(token):
        return ""
    return _BOUNDARY_PUNCT_RE.sub("", token)


def is_retainable_token(token: str) -> bool:
    """
    Decide whether a token survives numeric/punctuation filtering (Step 1).

    Pure numbers/punctuation are dropped UNLESS the token looks like a
    clinically meaningful code (e.g. l4-l5 for spine levels).
    """
    if not token:
        return False
    if _CLINICAL_CODE_RE.match(token):
        return True
    if _NO_ALPHA_RE.match(token):
        return False
    return True


def normalize_text(text: str) -> str:
    """
    Lowercase and normalize surgical abbreviations before tokenization (Step 1).

    Order matters and runs **before** phrase detection:
      1. drop side marker ``(S)`` / ``(s)``
      2. ``w/o`` / ``W/O`` -> whitespace (longer form first)
      3. ``w/`` / ``W/`` -> whitespace
      4. remaining ``/`` -> whitespace
      5. protect glued ``&`` compounds (``T&A``, ``A&P``) so NLTK will not
         split them into letters; a standalone ``&`` (spaces around it) is
         left alone and may act as a separator
    """
    text = str(text).lower().strip()

    # Side marker (S) — remove entirely (not a content word)
    text = _SIDE_MARKER_RE.sub(" ", text)

    # Slash abbreviations -> whitespace (case already folded)
    text = text.replace("w/o", " ")
    text = text.replace("w/", " ")
    text = text.replace("/", " ")

    # Keep letter/digit-bound ampersands intact via a temporary placeholder
    text = _AMP_BETWEEN_ALNUM_RE.sub(_AMP_PLACEHOLDER, text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def _restore_ampersand(token: str) -> str:
    """Undo the ampersand placeholder after NLTK tokenization."""
    return token.replace(_AMP_PLACEHOLDER, "&")


def tokenize_and_lemmatize(text: str) -> List[str]:
    """
    Sentence-tokenize -> word-tokenize -> POS-tag -> lemmatize -> baseline stopword
    filter for one document (Step 1).

    Expects ``text`` already passed through :func:`normalize_text`.
    Drops NLTK English stopwords plus ``EXTRA_BASELINE_STOPWORDS`` (with, wo,
    and, of, ...) so they never reach phrase detection.
    """
    stops = get_baseline_stopwords()
    tokens: List[str] = []
    for sentence in sent_tokenize(text):
        words = word_tokenize(sentence)
        tagged = pos_tag(words)
        for word, tag in tagged:
            cleaned = _restore_ampersand(word)
            cleaned = strip_boundary_punctuation(cleaned)
            cleaned = cleaned.lower()
            # Restore again in case placeholder straddled a boundary strip edge
            cleaned = _restore_ampersand(cleaned)
            if not is_retainable_token(cleaned):
                continue
            lemma = _LEMMATIZER.lemmatize(cleaned, pos=_penn_to_wordnet(tag))
            lemma = _restore_ampersand(lemma)
            if not is_retainable_token(lemma):
                continue
            if lemma in stops:
                continue
            tokens.append(lemma)
    return tokens


def filter_rows(
    df: pd.DataFrame,
    text_col: str,
    duration_col: str,
) -> Tuple[pd.DataFrame, dict]:
    """
    Drop rows with null/empty procedure text or null/non-positive duration.

    Step 1: logs how many rows were dropped and why.
    """
    n_start = len(df)
    stats = {
        "n_start": n_start,
        "dropped_null_empty_text": 0,
        "dropped_null_duration": 0,
        "dropped_nonpositive_duration": 0,
    }

    working = df.copy()
    text_series = working[text_col].astype(str).str.strip()
    null_or_empty = (
        working[text_col].isna()
        | text_series.eq("")
        | text_series.str.lower().eq("nan")
    )
    stats["dropped_null_empty_text"] = int(null_or_empty.sum())
    working = working.loc[~null_or_empty].copy()

    null_duration = working[duration_col].isna()
    stats["dropped_null_duration"] = int(null_duration.sum())
    working = working.loc[~null_duration].copy()

    duration_vals = pd.to_numeric(working[duration_col], errors="coerce")
    nonpositive = duration_vals.isna() | (duration_vals <= config.MIN_DURATION_MINUTES)
    stats["dropped_nonpositive_duration"] = int(nonpositive.sum())
    working = working.loc[~nonpositive].copy()
    working[duration_col] = duration_vals.loc[~nonpositive].astype(float)

    stats["n_end"] = len(working)
    stats["dropped_total"] = n_start - len(working)

    logger.info(
        "Row filtering: start=%d | drop empty text=%d | drop null duration=%d | "
        "drop non-positive duration=%d | end=%d",
        stats["n_start"],
        stats["dropped_null_empty_text"],
        stats["dropped_null_duration"],
        stats["dropped_nonpositive_duration"],
        stats["n_end"],
    )
    return working, stats


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = config.TEXT_COLUMN,
    duration_col: str = config.DURATION_COLUMN,
) -> Tuple[pd.DataFrame, List[List[str]]]:
    """
    Full Step 1 transform: filter -> clean -> tokenize/lemmatize.

    Adds ``case_id`` (stable join key) and ``cleaned_text`` (space-joined lemmas).
    Returns the working dataframe and the tokenized corpus (list of token lists).
    """
    if text_col not in df.columns:
        raise KeyError(
            f"Text column '{text_col}' not found. Available: {list(df.columns)}"
        )
    if duration_col not in df.columns:
        raise KeyError(
            f"Duration column '{duration_col}' not found. Available: {list(df.columns)}"
        )

    working, _ = filter_rows(df, text_col=text_col, duration_col=duration_col)
    working = working.reset_index(drop=True)
    working[config.CASE_ID_COLUMN] = working.index.astype(int)

    logger.info("Tokenizing and lemmatizing %d cases ...", len(working))
    corpus: List[List[str]] = []
    cleaned_strings: List[str] = []
    for raw in working[text_col].tolist():
        tokens = tokenize_and_lemmatize(normalize_text(raw))
        corpus.append(tokens)
        cleaned_strings.append(" ".join(tokens))

    working["cleaned_text"] = cleaned_strings
    working["n_tokens"] = [len(t) for t in corpus]

    empty_docs = sum(1 for t in corpus if len(t) == 0)
    if empty_docs:
        logger.warning(
            "%d cases produced zero tokens after cleaning; they remain in the "
            "dataframe but will contribute little to phrase / stopword stats.",
            empty_docs,
        )

    return working, corpus


def run(input_path=None) -> Tuple[pd.DataFrame, List[List[str]]]:
    """
    Execute Step 1 end-to-end and persist interim artifacts to disk.

    Artifacts
    ---------
    - ``data/interim/tokenized_corpus.pkl``
    - ``data/interim/cases_cleaned.parquet``
    """
    setup_logging()
    ensure_nltk_resources()
    ensure_dirs([config.DATA_INTERIM, config.DATA_PROCESSED, config.OUTPUTS])

    path = input_path or config.INPUT_FILE
    logger.info("Loading input: %s", path)
    df = load_table(path)
    logger.info("Loaded shape=%s", df.shape)

    working, corpus = preprocess_dataframe(df)

    keep_cols = [
        config.CASE_ID_COLUMN,
        config.TEXT_COLUMN,
        config.DURATION_COLUMN,
        "cleaned_text",
        "n_tokens",
    ]
    # Preserve optional EncounterID if present (handy for audits; not used as text)
    if "EncounterID" in working.columns:
        keep_cols.insert(1, "EncounterID")
    out_df = working[keep_cols].copy()

    save_pickle(corpus, config.TOKENIZED_CORPUS_PATH)
    out_df.to_parquet(config.CASES_CLEANED_PATH, index=False)

    logger.info("Wrote tokenized corpus -> %s", config.TOKENIZED_CORPUS_PATH)
    logger.info("Wrote cleaned cases    -> %s", config.CASES_CLEANED_PATH)
    logger.info(
        "Vocab size (unique tokens)=%d | mean tokens/case=%.1f",
        len({t for doc in corpus for t in doc}),
        float(sum(len(d) for d in corpus) / max(len(corpus), 1)),
    )
    return out_df, corpus


if __name__ == "__main__":
    run()
