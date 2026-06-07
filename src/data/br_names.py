"""Basketball-Reference player name to URL suffix mapping.

Uses the br_names.txt file bundled with basketball_reference_scraper
for exact name matching, with a fallback suffix builder.
"""

import os
import re
import unicodedata
from pathlib import Path

import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

_URLS: dict[str, str] = {}
_loaded = False


def _load_names():
    global _loaded, _URLS
    if _loaded:
        return _URLS
    try:
        import basketball_reference_scraper.players as brp
        path = Path(brp.__file__).parent / "br_names.txt"
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        _URLS[line.lower()] = line
    except (ImportError, Exception):
        pass
    _loaded = True
    return _URLS


def _build_basic_suffix(name: str) -> str:
    norm = unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode("utf-8")
    parts = norm.strip().split()
    if len(parts) < 2:
        return ""
    first = parts[0]
    last = parts[-1]
    initial = last[0].lower()
    last_part = last[:5].lower() if len(last) >= 5 else last.lower().ljust(5, "0")
    first_part = first[:2].lower() if len(first) >= 2 else first.lower().ljust(2, "0")
    return f"/players/{initial}/{last_part}{first_part}01.html"


def lookup_suffix(name: str) -> str | None:
    _load_names()
    normalized = name.lower().strip()

    if normalized in _URLS:
        canonical = _URLS[normalized]
        return _name_to_suffix(canonical)

    suffix = _build_basic_suffix(name)
    if suffix:
        try:
            r = requests.head(f"https://www.basketball-reference.com{suffix}", headers=HEADERS, timeout=5)
            if r.status_code == 200:
                _URLS[normalized] = name
                return suffix
        except Exception:
            pass
    return None


def _name_to_suffix(name: str) -> str | None:
    parts = name.strip().split()
    if len(parts) < 2:
        return None
    first = parts[0]
    last = parts[-1]
    initial = last[0].lower()
    last_part = last[:5].lower() if len(last) >= 5 else last.lower().ljust(5, "0")
    first_part = first[:2].lower() if len(first) >= 2 else first.lower().ljust(2, "0")
    return f"/players/{initial}/{last_part}{first_part}01.html"
