"""
Step 3 — Domain stopword candidate discovery (Sarica & Luo, 2021).

For every unique token (after phrase detection), compute:

1. TF(t)     — corpus relative frequency (high -> stopword-like)
2. IDF(t)    — log(N / DF(t)) (low -> stopword-like)
3. TFIDF(t)  — mean over docs containing t of
               (tf_in_doc * N / DF(t))  **without** log on the DF term
               (paper's harsher variant; not sklearn's default)
4. Entropy   — -Sum P(p|t) log P(p|t) over docs containing t
               (high -> spread evenly -> uninformative)

Top-N from each ranked list -> UNION -> candidate set.
Multi-word phrase tokens (containing ``_``) are filtered out before export —
they must never become stopword candidates.

This script **only proposes** candidates. It does not remove anything.
"""

from __future__ import annotations

import logging
import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from . import config
from .utils import (
    ensure_dirs,
    load_pickle,
    setup_logging,
    truncate_snippet,
)

logger = logging.getLogger("surgical_stopwords.discovery")


def _safe_log(x: float) -> float:
    """Natural log with a hard floor to avoid log(0)."""
    return math.log(max(x, 1e-12))


def compute_term_statistics(
    corpus: Sequence[Sequence[str]],
) -> pd.DataFrame:
    """
    Compute TF, IDF, paper-variant TF-IDF, and entropy for every token (Step 3).

    Each surgical case = one document (analogous to one patent in the paper).
    """
    n_docs = len(corpus)
    if n_docs == 0:
        raise ValueError("Empty corpus — cannot compute stopword statistics.")

    doc_lengths = [len(doc) for doc in corpus]
    total_tokens = sum(doc_lengths)
    if total_tokens == 0:
        raise ValueError("Corpus has zero tokens after phrase detection.")

    # token -> list of (doc_idx, count_in_doc) for docs that contain the token
    token_doc_counts: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    tf_global: Counter = Counter()

    for doc_idx, doc in enumerate(corpus):
        if not doc:
            continue
        counts = Counter(doc)
        for token, cnt in counts.items():
            tf_global[token] += cnt
            token_doc_counts[token].append((doc_idx, cnt))

    rows = []
    for token, raw_tf in tf_global.items():
        pairs = token_doc_counts[token]
        df = len(pairs)
        tf_score = raw_tf / total_tokens
        idf_score = _safe_log(n_docs / df)

        # Paper-variant TFIDF: mean over docs containing t of
        # (count_in_doc / doc_len) * (N / DF) — NO log on the DF term.
        tfidf_vals = []
        for doc_idx, cnt in pairs:
            doc_len = doc_lengths[doc_idx]
            if doc_len == 0:
                continue
            tfidf_vals.append((cnt / doc_len) * (n_docs / df))
        tfidf_score = float(np.mean(tfidf_vals)) if tfidf_vals else 0.0

        # Entropy H(t) = -Sum P(p|t) log P(p|t)
        entropy = 0.0
        for _, cnt in pairs:
            p = cnt / raw_tf
            entropy -= p * _safe_log(p)

        rows.append(
            {
                "token": token,
                "raw_tf": raw_tf,
                "df": df,
                "tf_score": tf_score,
                "idf_score": idf_score,
                "tfidf_score": tfidf_score,
                "entropy_score": entropy,
            }
        )

    return pd.DataFrame(rows)


def rank_and_select_candidates(
    stats: pd.DataFrame,
    top_n: int = config.STOPWORD_TOP_N,
) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Build four ranked lists and take their union as stopword candidates (Step 3).

    Ranking directions (stopword-like at the top of each list):
      TF descending, IDF ascending, TFIDF ascending, Entropy descending
    """
    # Scale down when the surgical vocab is small (paper used 2000 on millions
    # of patent terms). Cap at ~25% of vocab so the union stays reviewable.
    auto_cap = max(50, len(stats) // 4)
    n = min(top_n, len(stats), auto_cap)
    if n < top_n:
        logger.info(
            "Auto-scaled top-N from %d -> %d (vocab_size=%d, cap~vocab/4)",
            top_n,
            n,
            len(stats),
        )
    logger.info(
        "Selecting top-N=%d from vocab_size=%d (config STOPWORD_TOP_N=%d)",
        n,
        len(stats),
        top_n,
    )

    ranked = stats.copy()
    ranked["tf_rank"] = ranked["tf_score"].rank(ascending=False, method="min").astype(int)
    ranked["idf_rank"] = ranked["idf_score"].rank(ascending=True, method="min").astype(int)
    ranked["tfidf_rank"] = ranked["tfidf_score"].rank(ascending=True, method="min").astype(int)
    ranked["entropy_rank"] = (
        ranked["entropy_score"].rank(ascending=False, method="min").astype(int)
    )

    top_tf = set(ranked.nsmallest(n, "tf_rank")["token"])
    top_idf = set(ranked.nsmallest(n, "idf_rank")["token"])
    top_tfidf = set(ranked.nsmallest(n, "tfidf_rank")["token"])
    top_entropy = set(ranked.nsmallest(n, "entropy_rank")["token"])

    candidates = top_tf | top_idf | top_tfidf | top_entropy
    logger.info(
        "Union size before phrase filter: %d "
        "(tf=%d idf=%d tfidf=%d entropy=%d)",
        len(candidates),
        len(top_tf),
        len(top_idf),
        len(top_tfidf),
        len(top_entropy),
    )

    # Drop multi-word phrases glued by Step 2 — never stopword candidates
    with_underscore = {t for t in candidates if config.PHRASE_DELIMITER in t}
    if with_underscore:
        logger.info(
            "Dropping %d phrase candidates containing '%s' (e.g. %s)",
            len(with_underscore),
            config.PHRASE_DELIMITER,
            sorted(with_underscore)[:5],
        )
    candidates -= with_underscore

    ranked["in_tf_top"] = ranked["token"].isin(top_tf)
    ranked["in_idf_top"] = ranked["token"].isin(top_idf)
    ranked["in_tfidf_top"] = ranked["token"].isin(top_tfidf)
    ranked["in_entropy_top"] = ranked["token"].isin(top_entropy)
    ranked["n_metrics_flagged"] = (
        ranked["in_tf_top"].astype(int)
        + ranked["in_idf_top"].astype(int)
        + ranked["in_tfidf_top"].astype(int)
        + ranked["in_entropy_top"].astype(int)
    )

    cand_df = ranked[ranked["token"].isin(candidates)].copy()
    return cand_df, candidates


def build_candidate_export(
    cand_df: pd.DataFrame,
    cases: pd.DataFrame,
    corpus: Sequence[Sequence[str]],
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Assemble ``candidate_stopwords.csv`` schema for offline human review (Step 3/4).

    Columns include all four metric scores/ranks, overlap count, an example
    snippet, and empty ``human_decision`` / ``notes`` fields.
    """
    rng = random.Random(seed)

    # Map token -> list of case indices containing it (for random snippet)
    token_to_docs: Dict[str, List[int]] = defaultdict(list)
    for i, doc in enumerate(corpus):
        for token in set(doc):
            token_to_docs[token].append(i)

    snippets = []
    for token in cand_df["token"]:
        docs = token_to_docs.get(token, [])
        if not docs:
            snippets.append("")
            continue
        doc_idx = rng.choice(docs)
        # Prefer original procedure text for readable context
        if config.TEXT_COLUMN in cases.columns:
            raw = str(cases.iloc[doc_idx][config.TEXT_COLUMN])
        else:
            raw = " ".join(corpus[doc_idx])
        snippets.append(truncate_snippet(raw, config.EXAMPLE_SNIPPET_CHARS))

    export = pd.DataFrame(
        {
            "token": cand_df["token"].values,
            "tf_score": cand_df["tf_score"].values,
            "tf_rank": cand_df["tf_rank"].values,
            "idf_score": cand_df["idf_score"].values,
            "idf_rank": cand_df["idf_rank"].values,
            "tfidf_score": cand_df["tfidf_score"].values,
            "tfidf_rank": cand_df["tfidf_rank"].values,
            "entropy_score": cand_df["entropy_score"].values,
            "entropy_rank": cand_df["entropy_rank"].values,
            "n_metrics_flagged": cand_df["n_metrics_flagged"].values,
            "example_case_snippet": snippets,
            "human_decision": "",  # fill offline: keep | remove
            "notes": "",
        }
    )
    export = export.sort_values(
        by=["n_metrics_flagged", "tf_score"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return export


def run(top_n: int = None) -> pd.DataFrame:
    """
    Execute Step 3: score tokens, form candidate union, export CSV.

    Artifact
    --------
    - ``outputs/candidate_stopwords.csv``
    """
    setup_logging()
    ensure_dirs([config.OUTPUTS])

    corpus = load_pickle(config.PHRASE_CORPUS_PATH)
    cases = pd.read_parquet(config.CASES_CLEANED_PATH)
    if len(cases) != len(corpus):
        raise ValueError(
            f"Length mismatch: cases={len(cases)} vs phrase_corpus={len(corpus)}"
        )

    logger.info("Computing TF / IDF / TFIDF / entropy for phrase-joined corpus ...")
    stats = compute_term_statistics(corpus)
    logger.info("Vocabulary size: %d", len(stats))

    n = top_n if top_n is not None else config.STOPWORD_TOP_N
    cand_df, candidates = rank_and_select_candidates(stats, top_n=n)
    logger.info("Candidate stopwords after phrase filter: %d", len(candidates))

    export = build_candidate_export(cand_df, cases, corpus)
    export.to_csv(config.CANDIDATE_STOPWORDS_PATH, index=False)
    logger.info("Wrote candidates -> %s", config.CANDIDATE_STOPWORDS_PATH)
    logger.info(
        "Review next: open the CSV in Excel, set human_decision to "
        "'keep' or 'remove' for every row, save back to the same path."
    )
    return export


if __name__ == "__main__":
    run()
