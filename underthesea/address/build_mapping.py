"""
Script to extract mapping data from vietnamadminunits package
and generate data/mapping.json for standalone use.

Usage:
    uv run python scripts/build_mapping.py
"""

import json
from pathlib import Path


def build_mapping():
    import vietnamadminunits

    pkg_dir = Path(vietnamadminunits.__file__).parent

    # Load source data
    with open(pkg_dir / "data" / "converter_2025.json") as f:
        converter = json.load(f)
    with open(pkg_dir / "data" / "parser_legacy.json") as f:
        legacy = json.load(f)
    with open(pkg_dir / "data" / "parser_from_2025.json") as f:
        new_parser = json.load(f)

    # === Province mapping: old_key -> new_key ===
    # converter DICT_PROVINCE: {new_key: [old_key1, old_key2, ...]}
    province_mapping = {}
    for new_key, old_keys in converter["DICT_PROVINCE"].items():
        for old_key in old_keys:
            province_mapping[old_key] = new_key

    # === Province info: key -> display name ===
    province_names = {}
    for key, info in new_parser["DICT_PROVINCE"].items():
        province_names[key] = {
            "name": info["province"],
            "short": info["provinceShort"],
            "code": info["provinceCode"],
        }

    old_province_names = {}
    for key, info in legacy["DICT_PROVINCE"].items():
        old_province_names[key] = {
            "name": info["province"],
            "short": info["provinceShort"],
            "code": info["provinceCode"],
        }

    # === New ward info: province_key -> ward_key -> display name ===
    new_ward_names = {}
    for prov_key, wards in new_parser["DICT_PROVINCE_WARD_NO_ACCENTED"].items():
        new_ward_names[prov_key] = {}
        for ward_key, info in wards.items():
            new_ward_names[prov_key][ward_key] = {
                "name": info["ward"],
                "short": info["wardShort"],
                "type": info["wardType"],
                "code": info["wardCode"],
            }

    # === Old ward info: province_key -> district_key -> ward_key -> display name ===
    old_ward_names = {}
    for prov_key, districts in legacy["DICT_PROVINCE_DISTRICT_WARD_NO_ACCENTED"].items():
        old_ward_names[prov_key] = {}
        for dist_key, wards in districts.items():
            for ward_key, info in wards.items():
                old_ward_names[prov_key][f"{prov_key}_{dist_key}_{ward_key}"] = {
                    "name": info["ward"],
                    "short": info["wardShort"],
                    "type": info["wardType"],
                    "code": info["wardCode"],
                }

    # === Old district info ===
    old_district_names = {}
    for prov_key, districts in legacy.get("DICT_PROVINCE_DISTRICT", {}).items():
        old_district_names[prov_key] = {}
        for dist_key, info in districts.items():
            old_district_names[prov_key][dist_key] = {
                "name": info.get("district", ""),
                "short": info.get("districtShort", ""),
                "type": info.get("districtType", ""),
            }

    # === Ward mapping records ===
    ward_mapping = []

    # NO_DIVIDED: each new ward maps to one or more old wards (unchanged or renamed/merged)
    for new_prov_key, wards in converter["DICT_PROVINCE_WARD_NO_DIVIDED"].items():
        new_prov_info = province_names.get(new_prov_key, {})

        for new_ward_key, old_compound_keys in wards.items():
            new_ward_info = new_ward_names.get(new_prov_key, {}).get(new_ward_key, {})

            for old_compound_key in old_compound_keys:
                # Parse old compound key: "old_prov_key_old_dist_key_old_ward_key"
                parts = old_compound_key.split("_", 2)
                if len(parts) < 2:
                    continue
                old_prov_key = parts[0]
                rest = "_".join(parts[1:]) if len(parts) > 1 else ""

                # Find old ward info
                old_full_key = old_compound_key
                old_ward_info = {}
                old_dist_info = {}

                # Find in old_ward_names
                if old_prov_key in old_ward_names:
                    old_ward_info = old_ward_names[old_prov_key].get(old_full_key, {})

                # Parse district key from compound
                if len(parts) == 3:
                    old_dist_key = parts[1]
                    old_ward_key_str = parts[2]
                    if old_prov_key in old_district_names:
                        old_dist_info = old_district_names[old_prov_key].get(old_dist_key, {})
                elif len(parts) == 2:
                    old_dist_key = parts[1]
                    old_ward_key_str = ""
                    if old_prov_key in old_district_names:
                        old_dist_info = old_district_names[old_prov_key].get(old_dist_key, {})

                # Determine mapping type
                if len(old_compound_keys) == 1:
                    # Only one old ward maps to this new ward
                    if old_ward_info.get("name") == new_ward_info.get("name"):
                        mapping_type = "unchanged"
                    else:
                        mapping_type = "renamed"
                else:
                    mapping_type = "merged"

                record = {
                    "old_province": old_province_names.get(old_prov_key, {}).get("name", ""),
                    "old_province_key": old_prov_key,
                    "old_district": old_dist_info.get("name", ""),
                    "old_district_key": parts[1] if len(parts) >= 2 else "",
                    "old_ward": old_ward_info.get("name", ""),
                    "old_ward_key": old_ward_key_str if len(parts) == 3 else "",
                    "new_province": new_prov_info.get("name", ""),
                    "new_province_key": new_prov_key,
                    "new_ward": new_ward_info.get("name", ""),
                    "new_ward_key": new_ward_key,
                    "mapping_type": mapping_type,
                }
                ward_mapping.append(record)

    # DIVIDED: old wards split into multiple new wards
    for new_prov_key, old_wards in converter["DICT_PROVINCE_WARD_DIVIDED"].items():
        new_prov_info = province_names.get(new_prov_key, {})

        for old_compound_key, new_ward_options in old_wards.items():
            parts = old_compound_key.split("_", 2)
            if len(parts) < 2:
                continue
            old_prov_key = parts[0]

            old_ward_info = {}
            old_dist_info = {}
            if old_prov_key in old_ward_names:
                old_ward_info = old_ward_names[old_prov_key].get(old_compound_key, {})
            if len(parts) >= 2 and old_prov_key in old_district_names:
                old_dist_info = old_district_names[old_prov_key].get(parts[1], {})

            for option in new_ward_options:
                new_ward_key = option["newWardKey"]
                new_ward_info = new_ward_names.get(new_prov_key, {}).get(new_ward_key, {})

                record = {
                    "old_province": old_province_names.get(old_prov_key, {}).get("name", ""),
                    "old_province_key": old_prov_key,
                    "old_district": old_dist_info.get("name", ""),
                    "old_district_key": parts[1] if len(parts) >= 2 else "",
                    "old_ward": old_ward_info.get("name", ""),
                    "old_ward_key": parts[2] if len(parts) == 3 else "",
                    "new_province": new_prov_info.get("name", ""),
                    "new_province_key": new_prov_key,
                    "new_ward": new_ward_info.get("name", ""),
                    "new_ward_key": new_ward_key,
                    "mapping_type": "divided",
                    "is_default": option.get("isDefaultNewWard", False),
                }
                ward_mapping.append(record)

    # Build final mapping
    mapping = {
        "metadata": {
            "source": "vietnamadminunits",
            "version": "1.0.4",
            "effective_date": "2025-07-01",
            "old_provinces": len(old_province_names),
            "new_provinces": len(province_names),
            "total_records": len(ward_mapping),
        },
        "province_mapping": province_mapping,
        "province_names": province_names,
        "old_province_names": old_province_names,
        "ward_mapping": ward_mapping,
    }

    output = Path(__file__).parent.parent / "data" / "mapping.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"Generated {output}")
    print(f"  Province mappings: {len(province_mapping)} old -> {len(province_names)} new")
    print(f"  Ward mapping records: {len(ward_mapping)}")

    # Stats
    types = {}
    for r in ward_mapping:
        t = r["mapping_type"]
        types[t] = types.get(t, 0) + 1
    for t, c in sorted(types.items()):
        print(f"    {t}: {c}")


if __name__ == "__main__":
    build_mapping()
