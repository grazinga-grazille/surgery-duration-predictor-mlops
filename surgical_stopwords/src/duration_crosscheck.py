"""
Step 5 — Duration cross-check for human-approved removal candidates.

A token marked ``remove`` in ``candidate_stopwords.csv`` is only eligible for
the final stopword list if it does **not** show meaningful association with
``ActualDurationMinutes``. Human ``keep`` decisions are never overridden.

Association tests (binary presence of token vs duration):
  - Point-biserial correlation (Pearson between binary indicator and duration)
  - Mann-Whitney U (duration distribution with token present vs absent)

A token is flagged as having duration signal (blocked from removal) if either:
  |r_pb| >= DURATION_PBISER_ABS_THRESHOLD, or
  Mann-Whitney p < DURATION_MW_P_THRESHOLD
(with minimum group sizes enforced).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .utils import (
    ensure_dirs,
    load_pickle,
    normalize_human_decision,
    save_text_list,
    setup_logging,
)

logger = logging.getLogger("surgical_stopwords.duration")


def load_human_remove_candidates(path=None) -> pd.DataFrame:
    """
    Read ``candidate_stopwords.csv`` after offline review (Step 4 -> Step 5).

    Returns rows whose ``human_decision`` normalizes to ``remove``.
    Raises if any row is still undecided (empty decision).
    """
    path = path or config.CANDIDATE_STOPWORDS_PATH
    df = pd.read_csv(path)
    if "human_decision" not in df.columns:
        raise KeyError(
            f"{path} is missing 'human_decision'. Complete Step 4 review first."
        )

    df = df.copy()
    df["human_decision_norm"] = df["human_decision"].map(normalize_human_decision)

    undecided = df["human_decision_norm"].eq("") | ~df["human_decision_norm"].isin(
        {"keep", "remove"}
    )
    n_undecided = int(undecided.sum())
    if n_undecided:
        examples = df.loc[undecided, "token"].head(10).tolist()
        raise ValueError(
            f"{n_undecided} candidate(s) still lack a valid human_decision "
            f"('keep' or 'remove'). Examples: {examples}"
        )

    remove_df = df[df["human_decision_norm"] == "remove"].copy()
    keep_df = df[df["human_decision_norm"] == "keep"].copy()
    logger.info(
        "Human review loaded: remove=%d keep=%d total=%d",
        len(remove_df),
        len(keep_df),
        len(df),
    )
    return remove_df


def token_presence_matrix(
    corpus: Sequence[Sequence[str]],
    tokens: Sequence[str],
) -> Dict[str, np.ndarray]:
    """
    Build a binary presence vector (length = n_cases) for each token (Step 5).
    """
    token_set = set(tokens)
    vectors = {t: np.zeros(len(corpus), dtype=np.int8) for t in token_set}
    for i, doc in enumerate(corpus):
        present = token_set.intersection(doc)
        for t in present:
            vectors[t][i] = 1
    return vectors


def test_token_vs_duration(
    presence: np.ndarray,
    duration: np.ndarray,
    token: str,
) -> dict:
    """
    Point-biserial + Mann-Whitney tests for one token vs duration (Step 5).

    Returns a result dict including whether the token has duration signal.
    """
    n_present = int(presence.sum())
    n_absent = int(len(presence) - n_present)
    result = {
        "token": token,
        "n_present": n_present,
        "n_absent": n_absent,
        "mean_duration_present": np.nan,
        "mean_duration_absent": np.nan,
        "point_biserial_r": np.nan,
        "point_biserial_p": np.nan,
        "mannwhitney_u": np.nan,
        "mannwhitney_p": np.nan,
        "has_duration_signal": False,
        "safe_to_remove": False,
        "block_reason": "",
    }

    if n_present < config.DURATION_MIN_GROUP_SIZE:
        result["block_reason"] = (
            f"n_present={n_present} < min_group_size="
            f"{config.DURATION_MIN_GROUP_SIZE}; cannot confirm safety — blocked"
        )
        result["has_duration_signal"] = True  # conservative: do not remove
        result["safe_to_remove"] = False
        return result
    if n_absent < config.DURATION_MIN_GROUP_SIZE:
        result["block_reason"] = (
            f"n_absent={n_absent} < min_group_size="
            f"{config.DURATION_MIN_GROUP_SIZE}; cannot confirm safety — blocked"
        )
        result["has_duration_signal"] = True
        result["safe_to_remove"] = False
        return result

    y_present = duration[presence == 1]
    y_absent = duration[presence == 0]
    result["mean_duration_present"] = float(np.mean(y_present))
    result["mean_duration_absent"] = float(np.mean(y_absent))

    # Point-biserial == Pearson(binary, continuous)
    r, r_p = stats.pearsonr(presence.astype(float), duration.astype(float))
    result["point_biserial_r"] = float(r)
    result["point_biserial_p"] = float(r_p)

    u_stat, u_p = stats.mannwhitneyu(y_present, y_absent, alternative="two-sided")
    result["mannwhitney_u"] = float(u_stat)
    result["mannwhitney_p"] = float(u_p)

    signal_pb = abs(r) >= config.DURATION_PBISER_ABS_THRESHOLD
    signal_mw = u_p < config.DURATION_MW_P_THRESHOLD
    has_signal = bool(signal_pb or signal_mw)
    result["has_duration_signal"] = has_signal
    result["safe_to_remove"] = not has_signal

    reasons = []
    if signal_pb:
        reasons.append(
            f"|r_pb|={abs(r):.4f} >= {config.DURATION_PBISER_ABS_THRESHOLD}"
        )
    if signal_mw:
        reasons.append(f"MW p={u_p:.4g} < {config.DURATION_MW_P_THRESHOLD}")
    result["block_reason"] = "; ".join(reasons) if reasons else ""
    return result


def build_final_stopword_list(
    crosscheck: pd.DataFrame,
) -> List[str]:
    """
    Tokens that clear BOTH human ``remove`` and duration safety (Step 5).

    Human ``keep`` is never present in ``crosscheck`` (only remove candidates
    are tested). Safe-to-remove tokens become the final stopword list.
    """
    approved = crosscheck.loc[crosscheck["safe_to_remove"], "token"].tolist()
    blocked = crosscheck.loc[~crosscheck["safe_to_remove"], "token"].tolist()
    logger.info(
        "Duration gate: approved_for_removal=%d blocked=%d",
        len(approved),
        len(blocked),
    )
    if blocked:
        logger.info(
            "Blocked examples (have duration signal or insufficient n): %s",
            blocked[:15],
        )
    return sorted(set(approved))


def run() -> Tuple[pd.DataFrame, List[str]]:
    """
    Execute Step 5: duration cross-check + write final stopword list.

    Artifacts
    ---------
    - ``outputs/duration_crosscheck.csv``
    - ``outputs/final_stopwords.txt``
    """
    setup_logging()
    ensure_dirs([config.OUTPUTS])

    remove_df = load_human_remove_candidates()
    if remove_df.empty:
        logger.warning(
            "No tokens marked 'remove'. Writing empty final stopword list."
        )
        empty = pd.DataFrame(
            columns=[
                "token", "n_present", "n_absent",
                "mean_duration_present", "mean_duration_absent",
                "point_biserial_r", "point_biserial_p",
                "mannwhitney_u", "mannwhitney_p",
                "has_duration_signal", "safe_to_remove", "block_reason",
            ]
        )
        empty.to_csv(config.DURATION_CROSSCHECK_PATH, index=False)
        save_text_list([], config.FINAL_STOPWORDS_PATH)
        return empty, []

    corpus = load_pickle(config.PHRASE_CORPUS_PATH)
    cases = pd.read_parquet(config.CASES_CLEANED_PATH)
    duration = cases[config.DURATION_COLUMN].to_numpy(dtype=float)

    tokens = remove_df["token"].astype(str).tolist()
    vectors = token_presence_matrix(corpus, tokens)

    results = [
        test_token_vs_duration(vectors[token], duration, token) for token in tokens
    ]
    crosscheck = pd.DataFrame(results)
    # Attach human notes if present
    if "notes" in remove_df.columns:
        note_map = dict(zip(remove_df["token"].astype(str), remove_df["notes"]))
        crosscheck["human_notes"] = crosscheck["token"].map(note_map)

    crosscheck.to_csv(config.DURATION_CROSSCHECK_PATH, index=False)
    final_list = build_final_stopword_list(crosscheck)
    save_text_list(final_list, config.FINAL_STOPWORDS_PATH)

    logger.info("Wrote duration cross-check -> %s", config.DURATION_CROSSCHECK_PATH)
    logger.info("Wrote final stopwords (%d) -> %s", len(final_list), config.FINAL_STOPWORDS_PATH)
    return crosscheck, final_list


if __name__ == "__main__":
    run()
