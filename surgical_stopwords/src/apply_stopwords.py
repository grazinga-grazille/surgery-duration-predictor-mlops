"""
Step 6 — Apply the human-and-duration-vetted stopword list.

Reads ``outputs/final_stopwords.txt`` (produced only after Step 5 clears both
gates) and removes those tokens from the phrase-joined corpus. Phrase tokens
containing underscores are never in the final list (filtered in Step 3), so
multi-word surgical terms remain intact.
"""

from __future__ import annotations

import logging
from typing import List, Sequence, Set, Tuple

import pandas as pd

from . import config
from .utils import (
    ensure_dirs,
    load_pickle,
    load_text_list,
    save_pickle,
    setup_logging,
    tokens_to_text,
)

logger = logging.getLogger("surgical_stopwords.apply")


def filter_stopwords(
    corpus: Sequence[Sequence[str]],
    stopwords: Set[str],
) -> List[List[str]]:
    """
    Remove vetted stopword tokens from each document (Step 6).

    Order of remaining tokens is preserved.
    """
    if not stopwords:
        return [list(doc) for doc in corpus]
    return [[t for t in doc if t not in stopwords] for doc in corpus]


def run() -> Tuple[pd.DataFrame, List[List[str]]]:
    """
    Execute Step 6: apply final stopwords and persist cleaned artifacts.

    Artifacts
    ---------
    - ``data/processed/stopword_filtered_corpus.pkl``
    - ``data/processed/cases_final.parquet``
    """
    setup_logging()
    ensure_dirs([config.DATA_PROCESSED])

    stopwords = set(load_text_list(config.FINAL_STOPWORDS_PATH))
    logger.info("Loaded %d final stopwords", len(stopwords))

    corpus = load_pickle(config.PHRASE_CORPUS_PATH)
    cases = pd.read_parquet(config.CASES_CLEANED_PATH)
    if len(cases) != len(corpus):
        raise ValueError(
            f"Length mismatch: cases={len(cases)} vs phrase_corpus={len(corpus)}"
        )

    filtered = filter_stopwords(corpus, stopwords)
    n_removed = sum(
        len(a) - len(b) for a, b in zip(corpus, filtered)
    )
    logger.info(
        "Removed %d stopword token occurrences across %d cases",
        n_removed,
        len(filtered),
    )

    out = cases.copy()
    out["cleaned_text"] = [tokens_to_text(doc) for doc in filtered]
    out["n_tokens"] = [len(doc) for doc in filtered]

    save_pickle(filtered, config.STOPWORD_FILTERED_CORPUS_PATH)
    out.to_parquet(config.CASES_FINAL_PATH, index=False)

    logger.info("Wrote filtered corpus -> %s", config.STOPWORD_FILTERED_CORPUS_PATH)
    logger.info("Wrote final cases     -> %s", config.CASES_FINAL_PATH)
    return out, filtered


if __name__ == "__main__":
    run()
