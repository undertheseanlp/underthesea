"""Core address conversion logic."""

import json
from pathlib import Path

from .models import ConversionResult, ConversionStatus, MappingType
from .normalizer import normalize_for_matching, normalize_key
from .parser import parse_address

# Load mapping data
_DATA_PATH = Path(__file__).parent / "data" / "mapping.json"
_mapping_data = None
_index = None


def _load_data():
    global _mapping_data, _index
    if _mapping_data is not None:
        return

    with open(_DATA_PATH, encoding="utf-8") as f:
        _mapping_data = json.load(f)

    _index = _build_index(_mapping_data)


def _build_index(data: dict) -> dict:
    """Build lookup indices for fast matching."""
    index = {
        # old_province_key -> new_province_key
        "province": data["province_mapping"],
        # province_names for display
        "province_names": data["province_names"],
        "old_province_names": data["old_province_names"],
        # Exact match: (old_prov_key, old_dist_key, old_ward_key) -> list of records
        "exact": {},
        # Fuzzy: (old_prov_key, old_ward_key) -> list of records (ignoring district)
        "ward_only": {},
        # Province keyword lookup: normalized_name -> province_key
        "province_keywords": {},
    }

    # Build province keyword index
    for key, info in data["old_province_names"].items():
        index["province_keywords"][normalize_key(info["name"])] = key
        index["province_keywords"][normalize_key(info["short"])] = key
        index["province_keywords"][key] = key

    # Build ward indices
    for record in data["ward_mapping"]:
        prov_key = record["old_province_key"]
        dist_key = record["old_district_key"]
        ward_key = record["old_ward_key"]

        # Exact match index
        exact_key = (prov_key, dist_key, ward_key)
        index["exact"].setdefault(exact_key, []).append(record)

        # Ward-only index (for matching without district)
        wo_key = (prov_key, ward_key)
        index["ward_only"].setdefault(wo_key, []).append(record)

    return index


def _resolve_province(text: str) -> str | None:
    """Resolve a province string to its key."""
    normalized = normalize_for_matching(text)
    return _index["province_keywords"].get(normalized)


def _find_mapping(old_prov_key: str, old_dist_key: str, old_ward_key: str) -> list[dict]:
    """Find mapping records for given old admin unit keys."""
    # Tier 1: Exact match (province + district + ward)
    exact_key = (old_prov_key, old_dist_key, old_ward_key)
    records = _index["exact"].get(exact_key, [])
    if records:
        return records

    # Tier 2: Ward-only match (province + ward, ignoring district)
    wo_key = (old_prov_key, old_ward_key)
    records = _index["ward_only"].get(wo_key, [])
    if records:
        return records

    return []


def _select_best_record(records: list[dict]) -> dict | None:
    """Select the best record from multiple matches."""
    if not records:
        return None
    if len(records) == 1:
        return records[0]

    # For divided wards, prefer the default
    for r in records:
        if r.get("is_default"):
            return r

    # Otherwise return the first
    return records[0]


def convert_address(address: str) -> ConversionResult:
    """
    Convert a Vietnamese address from old format (63 provinces, 3-level)
    to new format (34 provinces, 2-level).

    Args:
        address: Vietnamese address string, e.g.
            "Phường Phúc Xá, Quận Ba Đình, Thành phố Hà Nội"

    Returns:
        ConversionResult with conversion details.
    """
    _load_data()

    result = ConversionResult(original=address)
    parsed = parse_address(address)
    result.old = parsed

    # Resolve province
    old_prov_key = _resolve_province(parsed.province)
    if not old_prov_key:
        # Try district field as province (2-part address might be misparse)
        if parsed.district:
            old_prov_key = _resolve_province(parsed.district)
        if not old_prov_key:
            result.status = ConversionStatus.NOT_FOUND
            result.note = f"Province not found: {parsed.province}"
            return result

    # Get new province
    new_prov_key = _index["province"].get(old_prov_key)
    if not new_prov_key:
        result.status = ConversionStatus.NOT_FOUND
        result.note = f"No province mapping for: {old_prov_key}"
        return result

    new_prov_info = _index["province_names"].get(new_prov_key, {})
    result.new.province = new_prov_info.get("name", "")

    # If no ward info, return province-only result
    if not parsed.ward and not parsed.district:
        result.status = ConversionStatus.PARTIAL
        result.converted = result.new.province
        result.note = "Province-only conversion"
        return result

    # Resolve ward
    old_dist_key = normalize_key(parsed.district) if parsed.district else ""
    old_ward_key = normalize_key(parsed.ward) if parsed.ward else ""

    records = _find_mapping(old_prov_key, old_dist_key, old_ward_key)

    if not records and parsed.ward:
        # Try ward in district field (for 2-part: "ward, province")
        old_ward_key2 = normalize_key(parsed.district) if parsed.district else ""
        if old_ward_key2:
            records = _find_mapping(old_prov_key, "", old_ward_key2)

    if not records:
        result.status = ConversionStatus.PARTIAL
        result.new.street = parsed.street
        result.converted = result.new.to_address()
        result.note = "Ward not found, province converted"
        return result

    record = _select_best_record(records)
    result.mapping_type = MappingType(record["mapping_type"])
    result.new.ward = record["new_ward"]
    result.new.street = parsed.street
    result.converted = result.new.to_address()
    result.status = ConversionStatus.SUCCESS

    if result.mapping_type == MappingType.DIVIDED:
        result.note = "Old ward was split; default new ward selected"

    return result


def batch_convert(addresses: list[str]) -> list[ConversionResult]:
    """Convert a list of addresses."""
    _load_data()
    return [convert_address(addr) for addr in addresses]
