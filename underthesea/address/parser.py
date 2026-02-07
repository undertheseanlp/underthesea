"""Parse Vietnamese address strings into components."""

from .models import AdminUnit
from .normalizer import expand_abbreviations

# Ward-level prefixes
WARD_PREFIXES = ("phường", "xã", "thị trấn")


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
