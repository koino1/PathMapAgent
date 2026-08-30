from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List


DATA_DIR = Path(__file__).resolve().parents[1] / "local_data"


def load_csv_rows(filename: str) -> List[Dict[str, Any]]:
    path = DATA_DIR / filename
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [normalize_row(row) for row in reader]


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in row.items():
        clean_key = str(key).replace("\ufeff", "").strip()
        if isinstance(value, str):
            normalized[clean_key] = value.strip()
        else:
            normalized[clean_key] = value
    return normalized


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def compact_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for record in records:
        cleaned.append(
            {
                key: value
                for key, value in record.items()
                if value not in ("", None, [], {})
            }
        )
    return cleaned


def dedupe_records(records: Iterable[Dict[str, Any]], keys: Iterable[str]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    key_list = list(keys)
    for record in records:
        marker = tuple(record.get(key) for key in key_list)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(record)
    return deduped
