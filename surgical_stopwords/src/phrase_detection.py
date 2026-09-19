"""
Step 2 — Multi-word phrase detection with gensim Phrases / Phraser.

Must run **before** stopword scoring so multi-word surgical terms
(e.g. ``total_hip_arthroplasty``) are single vocabulary items. Otherwise
the stopword scorer would evaluate ``total``, ``hip``, and ``arthroplasty``
as independent unigrams and may falsely flag common components.

Implements the Mikolov et al. (2013) PMI-style score used by gensim and
referenced by Sarica & Luo (2021):

    score(w_i, w_j) = (count(w_i, w_j) - delta) * N / (count(w_i) * count(w_j))

Two passes (paper approach):
  Pass 1 — higher threshold -> frequent, obvious bigrams
  Pass 2 — lower threshold on pass-1 output -> trigrams / four-grams
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Iterable, List, Sequence, Tuple

import pandas as pd
from gensim.models.phrases import Phraser, Phrases

from . import config
from .utils import ensure_dirs, load_pickle, save_pickle, setup_logging, tokens_to_text

logger = logging.getLogger("surgical_stopwords.phrases")


def train_two_pass_phrases(
    corpus: Sequence[Sequence[str]],
    min_count: int = config.PHRASE_MIN_COUNT,
    threshold_pass1: float = config.PHRASE_THRESHOLD_PASS1,
    threshold_pass2: float = config.PHRASE_THRESHOLD_PASS2,
) -> Tuple[Phrases, Phrases, Phraser, List[List[str]]]:
    """
    Train two-pass gensim Phrases models and return the final phrase-joined corpus.

    Step 2: pass-1 Phraser is applied before training pass-2 so higher-order
    n-grams can form from bigram combinations.
    """
    logger.info(
        "Phrase pass 1: min_count=%s threshold=%s (stricter bigrams)",
        min_count,
        threshold_pass1,
    )
    phrases1 = Phrases(
        corpus,
        min_count=min_count,
        threshold=threshold_pass1,
        delimiter=config.PHRASE_DELIMITER,
        scoring="default",
    )
    phraser1 = Phraser(phrases1)
    corpus_bigrams = [phraser1[doc] for doc in corpus]

    logger.info(
        "Phrase pass 2: min_count=%s threshold=%s (looser higher-order n-grams)",
        min_count,
        threshold_pass2,
    )
    phrases2 = Phrases(
        corpus_bigrams,
        min_count=min_count,
        threshold=threshold_pass2,
        delimiter=config.PHRASE_DELIMITER,
        scoring="default",
    )
    phraser2 = Phraser(phrases2)
    corpus_phrases = [phraser2[doc] for doc in corpus_bigrams]

    # Freeze a single reusable Phraser that applies both transforms in order
    # by storing the pass-2 model (which already saw bigram-joined input).
    # Downstream reuse: apply phraser1 then phraser2, or use saved corpus.
    return phrases1, phrases2, phraser2, corpus_phrases


def extract_phrase_frequencies(
    corpus: Sequence[Sequence[str]],
    delimiter: str = config.PHRASE_DELIMITER,
) -> pd.DataFrame:
    """
    Count multi-word phrase tokens (those containing the delimiter).

    Step 2: used to print/log the top 50 phrases for sanity-checking quality.
    """
    counter: Counter = Counter()
    for doc in corpus:
        for token in doc:
            if delimiter in token:
                counter[token] += 1
    rows = [
        {"phrase": phrase, "count": count}
        for phrase, count in counter.most_common()
    ]
    return pd.DataFrame(rows)


def apply_phrasers(
    corpus: Sequence[Sequence[str]],
    phraser1: Phraser,
    phraser2: Phraser,
) -> List[List[str]]:
    """Apply both frozen phrasers in order to a tokenized corpus (Step 2 reuse)."""
    return [phraser2[phraser1[doc]] for doc in corpus]


def run() -> List[List[str]]:
    """
    Execute Step 2 end-to-end from interim tokenized corpus.

    Artifacts
    ---------
    - ``data/interim/phrases_pass1.pkl`` / ``phrases_pass2.pkl``
    - ``data/interim/phraser.pkl`` (dict with phraser1 + phraser2)
    - ``data/interim/phrase_corpus.pkl``
    - ``data/interim/phrase_frequencies.csv``
    - Updates ``cleaned_text`` in ``cases_cleaned.parquet`` to phrase-joined text
    """
    setup_logging()
    ensure_dirs([config.DATA_INTERIM])

    corpus = load_pickle(config.TOKENIZED_CORPUS_PATH)
    logger.info("Loaded tokenized corpus: %d documents", len(corpus))

    phrases1, phrases2, _, corpus_phrases = train_two_pass_phrases(corpus)
    phraser1 = Phraser(phrases1)
    phraser2 = Phraser(phrases2)

    save_pickle(phrases1, config.PHRASE_MODEL_PASS1_PATH)
    save_pickle(phrases2, config.PHRASE_MODEL_PASS2_PATH)
    save_pickle({"phraser1": phraser1, "phraser2": phraser2}, config.PHRASER_PATH)
    save_pickle(corpus_phrases, config.PHRASE_CORPUS_PATH)

    freq_df = extract_phrase_frequencies(corpus_phrases)
    freq_df.to_csv(config.PHRASE_FREQ_PATH, index=False)

    top_n = min(50, len(freq_df))
    logger.info("Detected %d unique multi-word phrases", len(freq_df))
    logger.info("Top %d most frequent phrases (sanity check):", top_n)
    for _, row in freq_df.head(top_n).iterrows():
        logger.info("  %6d  %s", int(row["count"]), row["phrase"])

    # Keep cases dataframe aligned: refresh cleaned_text with phrase tokens
    cases = pd.read_parquet(config.CASES_CLEANED_PATH)
    if len(cases) != len(corpus_phrases):
        raise ValueError(
            f"Length mismatch: cases={len(cases)} vs phrase_corpus={len(corpus_phrases)}"
        )
    cases["cleaned_text"] = [tokens_to_text(doc) for doc in corpus_phrases]
    cases["n_tokens"] = [len(doc) for doc in corpus_phrases]
    cases.to_parquet(config.CASES_CLEANED_PATH, index=False)

    logger.info("Wrote phrase corpus -> %s", config.PHRASE_CORPUS_PATH)
    logger.info("Wrote phrase freqs  -> %s", config.PHRASE_FREQ_PATH)
    logger.info("Updated cases       -> %s", config.CASES_CLEANED_PATH)
    return corpus_phrases


if __name__ == "__main__":
    run()
