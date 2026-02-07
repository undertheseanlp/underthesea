"""Vietnamese text normalization for address matching.

Uses underthesea character normalization (NFC + character map) to fix
encoding issues, then applies address-specific transformations
(abbreviation expansion, diacritics removal, key generation).
"""

import re
import unicodedata

from underthesea.pipeline.text_normalize.character_normalize import (
    normalize_characters_in_text,
)

# Abbreviation expansions
ABBREVIATIONS = {
    "tp.": "thành phố ",
    "tp ": "thành phố ",
    "t.p.": "thành phố ",
    "t.p ": "thành phố ",
    "p.": "phường ",
    "q.": "quận ",
    "h.": "huyện ",
    "tx.": "thị xã ",
    "t.x.": "thị xã ",
    "tt.": "thị trấn ",
    "t.t.": "thị trấn ",
    "x.": "xã ",
}


def remove_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics from text.

    First applies underthesea character normalization (NFC + character map)
    to fix encoding issues, then strips combining marks via NFKD decomposition.
    """
    text = normalize_characters_in_text(text)
    nfkd = unicodedata.normalize("NFKD", text)
    result = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Handle đ/Đ separately (not decomposed by NFKD)
    result = result.replace("đ", "d").replace("Đ", "D")
    return result


def normalize_key(text: str) -> str:
    """Normalize text to a lookup key (lowercase, no diacritics, no spaces/punctuation)."""
    text = text.lower().strip()
    text = remove_diacritics(text)
    text = re.sub(r"[^a-z0-9]", "", text)
    return text


def expand_abbreviations(text: str) -> str:
    """Expand common Vietnamese address abbreviations."""
    result = text.lower().strip()
    # Sort by length descending to match longer abbreviations first
    for abbr, full in sorted(ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
        result = result.replace(abbr, full)
    return result.strip()


def normalize_for_matching(text: str) -> str:
    """Full normalization pipeline for fuzzy matching."""
    text = expand_abbreviations(text)
    return normalize_key(text)
