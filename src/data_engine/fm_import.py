"""
Import Football Manager exports (user-generated; not shipped with the game).

Supported:
  - FM26 Player Export plugin CSV (semicolon-delimited)
  - FM HTML squad export (table parse, lightweight)
"""

from __future__ import annotations

import csv
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional


def _norm_key(k: str) -> str:
    return re.sub(r"\s+", "_", k.strip().lower())


def parse_fm_csv(path: str | Path, *, delimiter: str = ";") -> List[Dict[str, Any]]:
    """
    Parse FM CSV export. Column names vary by view; common: Name, Age, Position, CA, PA, Nation.
    """
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        # sniff delimiter
        sample = f.read(4096)
        f.seek(0)
        if delimiter == ";" and sample.count(",") > sample.count(";") * 2:
            delimiter = ","
        reader = csv.DictReader(f, delimiter=delimiter)
        for raw in reader:
            row = {_norm_key(k): (v.strip() if isinstance(v, str) else v) for k, v in raw.items() if k}
            if not row:
                continue
            name = row.get("name") or row.get("player") or row.get("player_name")
            if not name:
                continue
            rows.append(_normalize_fm_row(row, name))
    return rows


def _parse_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    s = str(val).strip().replace(",", ".")
    if not s:
        return default
    try:
        return int(float(s))
    except ValueError:
        return default


def _parse_float01(val: Any, scale: float = 20.0) -> float:
    """Map FM 1–20 attribute or 0–100 CA to 0–1."""
    if val is None or str(val).strip() == "":
        return 0.5
    try:
        x = float(str(val).replace(",", "."))
    except ValueError:
        return 0.5
    if x > 1.5:
        x = x / scale
    return float(max(0.0, min(1.0, x)))


def _normalize_fm_row(row: Dict[str, Any], name: str) -> Dict[str, Any]:
    ca = row.get("ca") or row.get("current_ability") or row.get("cur")
    pa = row.get("pa") or row.get("potential_ability") or row.get("pot")
    nation = row.get("nation") or row.get("nationality") or row.get("nat")
    club = row.get("club") or row.get("team")
    pos = row.get("position") or row.get("pos") or row.get("preferred_position")

    abilities = {}
    for attr in (
        "pace",
        "acceleration",
        "stamina",
        "strength",
        "passing",
        "technique",
        "vision",
        "decisions",
        "composure",
        "finishing",
        "heading",
        "tackling",
        "positioning",
        "anticipation",
        "agility",
        "reflexes",
        "handling",
        "aerial_reach",
    ):
        if attr in row:
            abilities[attr] = _parse_float01(row[attr], scale=20.0)

    return {
        "name": name,
        "age": _parse_int(row.get("age")),
        "position": pos or "",
        "nation": nation or "",
        "club": club or "",
        "ca": _parse_int(ca),
        "pa": _parse_int(pa),
        "abilities_fm": abilities,
        "source": "fm_csv",
        "raw_keys": list(row.keys()),
    }


class _FMTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self._row: List[str] = []
        self._cell = ""
        self._in_td = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in ("td", "th"):
            self._in_td = True
            self._cell = ""

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._in_td:
            self._row.append(self._cell.strip())
            self._in_td = False
        elif tag == "tr":
            if self._row:
                self.rows.append(self._row)
            self._row = []

    def handle_data(self, data: str) -> None:
        if self._in_td:
            self._cell += data


def parse_fm_html(path: str | Path) -> List[Dict[str, Any]]:
    """Parse FM 'Web Page' squad export (first HTML table)."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    parser = _FMTableParser()
    parser.feed(text)
    if len(parser.rows) < 2:
        return []
    header = [_norm_key(h) for h in parser.rows[0]]
    out: List[Dict[str, Any]] = []
    for r in parser.rows[1:]:
        if len(r) != len(header):
            continue
        row = dict(zip(header, r))
        name = row.get("name") or row.get("player")
        if name:
            out.append(_normalize_fm_row(row, name))
    return out


def filter_players_by_nation(players: List[Dict[str, Any]], nation: str) -> List[Dict[str, Any]]:
    n = nation.lower().strip()
    return [p for p in players if n in str(p.get("nation", "")).lower()]
