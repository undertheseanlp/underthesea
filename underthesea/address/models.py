"""Data models for address conversion."""

from dataclasses import dataclass, field
from enum import Enum


class MappingType(str, Enum):
    UNCHANGED = "unchanged"
    RENAMED = "renamed"
    MERGED = "merged"
    DIVIDED = "divided"


class ConversionStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"  # Only province matched
    NOT_FOUND = "not_found"


@dataclass
class AdminUnit:
    province: str = ""
    district: str = ""
    ward: str = ""
    street: str = ""

    def to_address(self) -> str:
        parts = [p for p in (self.street, self.ward, self.district, self.province) if p]
        return ", ".join(parts)


@dataclass
class ConversionResult:
    original: str = ""
    converted: str = ""
    status: ConversionStatus = ConversionStatus.NOT_FOUND
    mapping_type: MappingType | None = None
    old: AdminUnit = field(default_factory=AdminUnit)
    new: AdminUnit = field(default_factory=AdminUnit)
    note: str = ""
