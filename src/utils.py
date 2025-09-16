# src/utils.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Dict, Any
import pandas as pd
import json


# ---------- Paths ----------
ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
FIG_DIR: Path = ROOT / "figures"
REPORT_DIR: Path = ROOT / "report"

__all__ = [
    "ROOT", "DATA_DIR", "RAW_DIR", "INTERIM_DIR", "PROCESSED_DIR",
    "FIG_DIR", "REPORT_DIR",
    "ensure_dirs",
    "read_raw", "read_processed",
    "save_df", "save_json", "load_json"
]


# ---------- IO Helpers ----------
def ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIG_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def read_raw(name: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Read a CSV from data/raw (or absolute/relative Path).
    """
    p = Path(name)
    if not p.is_absolute():
        p = RAW_DIR / p
    return _read_csv_safe(p, **kwargs)


def read_processed(name: Union[str, Path] = "netflix_clean.csv", **kwargs) -> pd.DataFrame:
    """
    Read a CSV from data/processed.
    """
    p = Path(name)
    if not p.is_absolute():
        p = PROCESSED_DIR / p
    return _read_csv_safe(p, **kwargs)


def save_df(df: pd.DataFrame, name: Union[str, Path], folder: Optional[Path] = None) -> Path:
    """
    Save DataFrame by extension:
      - .csv  -> CSV UTF-8
      - .xlsx -> Excel (requires openpyxl)
    Default folder:
      - figures/ when saving tables for quick reuse in report (e.g., 'top_countries.csv')
      - processed/ when saving cleaned datasets (e.g., 'netflix_clean.csv')
    """
    p = Path(name)
    if not p.suffix:
        p = p.with_suffix(".csv")

    if folder is None:
        folder = PROCESSED_DIR if p.name.endswith("_clean.csv") else FIG_DIR

    folder.mkdir(parents=True, exist_ok=True)
    out = folder / p.name

    if out.suffix.lower() == ".csv":
        df.to_csv(out, index=False)
    elif out.suffix.lower() in (".xlsx", ".xls"):
        try:
            df.to_excel(out, index=False)
        except Exception as e:
            raise RuntimeError("Saving .xlsx requires 'openpyxl'. Install it or save as .csv.") from e
    else:
        raise ValueError(f"Unsupported extension: {out.suffix}")

    return out


def save_json(obj: Dict[str, Any], name: Union[str, Path], folder: Optional[Path] = None, indent: int = 2) -> Path:
    """
    Save a dict as JSON (UTF-8). Defaults to report/ for configs/metadata.
    """
    p = Path(name)
    if not p.suffix:
        p = p.with_suffix(".json")
    if folder is None:
        folder = REPORT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / p.name
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    return out


def load_json(name: Union[str, Path], folder: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load JSON from report/ by default.
    """
    p = Path(name)
    if not p.is_absolute():
        base = REPORT_DIR if folder is None else folder
        p = base / p
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
