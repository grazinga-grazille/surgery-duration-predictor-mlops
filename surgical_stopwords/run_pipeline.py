#!/usr/bin/env python3
"""
Orchestrate the surgical stopword -> TF-IDF pipeline via CLI flags.

Examples
--------
    python run_pipeline.py --step preprocess
    python run_pipeline.py --step phrases
    python run_pipeline.py --step discover_stopwords
    python run_pipeline.py --step duration_crosscheck
    python run_pipeline.py --step apply_stopwords
    python run_pipeline.py --step build_embeddings
    python run_pipeline.py --step all_until_review   # Steps 1-3 only
    python run_pipeline.py --step all_after_review   # Steps 5-7 after Excel review
    python run_pipeline.py --step all                # full run (needs review file)

Step 4 (human review) is intentionally external — fill
``outputs/candidate_stopwords.csv`` offline, then continue.

Note: Run from the surgical_stopwords/ directory, or pass --input to override
the raw data path.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow `python run_pipeline.py` from surgical_stopwords/ without install
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config  # noqa: E402
from src.utils import ensure_dirs, setup_logging  # noqa: E402

logger = logging.getLogger("surgical_stopwords.pipeline")

STEP_CHOICES = [
    "preprocess",
    "phrases",
    "discover_stopwords",
    "duration_crosscheck",
    "apply_stopwords",
    "build_embeddings",
    "all_until_review",
    "all_after_review",
    "all",
]


def _banner(msg: str) -> None:
    logger.info("=" * 72)
    logger.info(msg)
    logger.info("=" * 72)


def run_preprocess(input_file: str = None) -> None:
    from src.preprocess import run as preprocess_run

    _banner("STEP 1 — Preprocess (clean / tokenize / lemmatize)")
    preprocess_run(input_path=input_file)


def run_phrases() -> None:
    from src.phrase_detection import run as phrases_run

    _banner("STEP 2 — Phrase detection (gensim two-pass Phrases)")
    phrases_run()


def run_discover(top_n: int = None) -> None:
    from src.stopword_discovery import run as discover_run

    _banner("STEP 3 — Stopword candidate discovery (Sarica & Luo 2021)")
    discover_run(top_n=top_n)


def run_duration() -> None:
    from src.duration_crosscheck import run as duration_run

    _banner("STEP 5 — Duration cross-check (point-biserial + Mann-Whitney)")
    duration_run()


def run_apply() -> None:
    from src.apply_stopwords import run as apply_run

    _banner("STEP 6 — Apply vetted stopwords")
    apply_run()


def run_embeddings() -> None:
    from src.build_embeddings import run as embed_run

    _banner("STEP 7 — Build TF-IDF embeddings")
    embed_run()


def run_all_until_review(input_file: str = None, top_n: int = None) -> None:
    """Steps 1-3; pause for offline human review (Step 4)."""
    run_preprocess(input_file=input_file)
    run_phrases()
    run_discover(top_n=top_n)
    _banner(
        "PAUSE — Step 4 human review\n"
        f"Open {config.CANDIDATE_STOPWORDS_PATH}\n"
        "Fill human_decision with 'keep' or 'remove' for every row, save, then run:\n"
        "  python run_pipeline.py --step all_after_review"
    )


def run_all_after_review() -> None:
    """Steps 5-7 after candidate_stopwords.csv has been reviewed."""
    run_duration()
    run_apply()
    run_embeddings()
    _banner("DONE — Steps 5-7 complete. TF-IDF artifacts are in data/processed/")


def run_all(input_file: str = None, top_n: int = None) -> None:
    """
    Full pipeline. Requires ``candidate_stopwords.csv`` to already contain
    completed human_decision values if you expect Steps 5-7 to succeed in the
    same invocation after a fresh Step 3 overwrite — typically you want
    ``all_until_review`` then ``all_after_review`` instead.
    """
    run_all_until_review(input_file=input_file, top_n=top_n)
    # Only continue if review appears complete
    import pandas as pd
    from src.utils import normalize_human_decision

    path = config.CANDIDATE_STOPWORDS_PATH
    if not path.exists():
        logger.error("Candidates CSV missing; cannot continue past review.")
        return
    df = pd.read_csv(path)
    decisions = df["human_decision"].map(normalize_human_decision)
    if decisions.eq("").any() or (~decisions.isin(["keep", "remove"])).any():
        logger.warning(
            "human_decision incomplete — stopping after Step 3. "
            "Finish Excel review, then: python run_pipeline.py --step all_after_review"
        )
        return
    run_all_after_review()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Surgical procedure text -> domain stopword discovery -> TF-IDF embeddings"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--step",
        required=True,
        choices=STEP_CHOICES,
        help="Which pipeline stage(s) to run",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Override path to raw CSV/Excel (default: config.INPUT_FILE)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Override STOPWORD_TOP_N for discover_stopwords",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging()
    ensure_dirs(
        [
            config.DATA_RAW,
            config.DATA_INTERIM,
            config.DATA_PROCESSED,
            config.OUTPUTS,
        ]
    )

    logger.info("Project root: %s", config.PROJECT_ROOT)
    logger.info("Running step: %s", args.step)

    dispatch = {
        "preprocess": lambda: run_preprocess(args.input),
        "phrases": run_phrases,
        "discover_stopwords": lambda: run_discover(args.top_n),
        "duration_crosscheck": run_duration,
        "apply_stopwords": run_apply,
        "build_embeddings": run_embeddings,
        "all_until_review": lambda: run_all_until_review(args.input, args.top_n),
        "all_after_review": run_all_after_review,
        "all": lambda: run_all(args.input, args.top_n),
    }
    dispatch[args.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
