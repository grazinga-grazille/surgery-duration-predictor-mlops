"""
Shared helpers for I/O, logging, and common data operations.

Used across Steps 1-7 of the surgical stopword / TF-IDF pipeline.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Union

import pandas as pd


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure root logging once and return a module-friendly logger.

    Step 0 / orchestration: keeps INFO-level progress visible when running
    on large surgical datasets.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    return logging.getLogger("surgical_stopwords")


def ensure_dirs(paths: Iterable[Path]) -> None:
    """Create parent directories for each path if they do not already exist."""
    for path in paths:
        path = Path(path)
        target = path if path.suffix == "" else path.parent
        target.mkdir(parents=True, exist_ok=True)


def load_table(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a CSV or Excel table, auto-detecting format from the file extension.

    Step 1: supports both .csv and .xlsx/.xls inputs without a separate flag.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path)
    raise ValueError(
        f"Unsupported input extension '{suffix}'. Use .csv, .xlsx, .xls, or .xlsm."
    )


def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    """Serialize ``obj`` to ``path`` with pickle (interim corpus / models)."""
    path = Path(path)
    ensure_dirs([path])
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Union[str, Path]) -> Any:
    """Load a pickle artifact written by :func:`save_pickle`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def save_text_list(items: Sequence[str], path: Union[str, Path]) -> None:
    """
    Write one string per line (e.g. final stopword list, TF-IDF feature names).

    Step 6 / 7 outputs.
    """
    path = Path(path)
    ensure_dirs([path])
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")


def load_text_list(path: Union[str, Path]) -> List[str]:
    """Read a one-token-per-line text file, skipping blank lines."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required text list not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def tokens_to_text(tokens: Sequence[str]) -> str:
    """Join a token list into a whitespace-separated string for TF-IDF input."""
    return " ".join(tokens)


def truncate_snippet(text: str, max_chars: int = 80) -> str:
    """Return a single-line truncated snippet for human review CSVs."""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1] + "..."


def normalize_human_decision(value: Optional[str]) -> str:
    """
    Normalize free-text human_decision cells to ``keep``, ``remove``, or ````.

    Step 5 reads decisions filled offline in Excel; be tolerant of casing
    and surrounding whitespace.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower()
    if text in {"keep", "k", "retain", "yes_keep"}:
        return "keep"
    if text in {"remove", "r", "drop", "stop", "stopword", "yes_remove"}:
        return "remove"
    return text
