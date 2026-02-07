"""Parse Vietnamese address strings into components."""

import re

from .models import AdminUnit
from .normalizer import expand_abbreviations

# Province-level prefixes
PROVINCE_PREFIXES = ("thành phố", "tỉnh")
# District-level prefixes
DISTRICT_PREFIXES = ("quận", "huyện", "thị xã", "thành phố")
# Ward-level prefixes
WARD_PREFIXES = ("phường", "xã", "thị trấn")


def _classify_part(text: str) -> str | None:
    """Classify an address part as province/district/ward/street."""
    lower = text.lower().strip()
    # Check province
    for prefix in PROVINCE_PREFIXES:
        if lower.startswith(prefix):
            # "Thành phố" can be both province and district
            # Province-level: TP Hà Nội, TP Hồ Chí Minh, etc.
            return None  # ambiguous, resolved by position
    # Check district
    for prefix in DISTRICT_PREFIXES:
        if lower.startswith(prefix):
            return None  # ambiguous
    # Check ward
    for prefix in WARD_PREFIXES:
        if lower.startswith(prefix):
            return "ward"
    return None


def parse_address(address: str) -> AdminUnit:
    """
    Parse Vietnamese address string into AdminUnit components.

    Expected format: "street, ward, district, province"
    Parsing is right-to-left (province is rightmost).
    """
    # Expand abbreviations first
    expanded = expand_abbreviations(address)

    # Split by comma
    parts = [p.strip() for p in expanded.split(",") if p.strip()]

    if not parts:
        return AdminUnit()

    unit = AdminUnit()

    # Right-to-left assignment
    if len(parts) >= 1:
        unit.province = parts[-1].strip()
    if len(parts) >= 2:
        unit.district = parts[-2].strip()
    if len(parts) >= 3:
        unit.ward = parts[-3].strip()
    if len(parts) >= 4:
        # Everything before ward is street
        unit.street = ", ".join(parts[:-3]).strip()

    # Handle 2-part addresses: could be "ward, province" or "district, province"
    if len(parts) == 2:
        lower = parts[0].lower().strip()
        for prefix in WARD_PREFIXES:
            if lower.startswith(prefix):
                unit.ward = unit.district
                unit.district = ""
                break

    return unit
