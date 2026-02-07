# convert_address

Convert Vietnamese addresses from the old 63-province 3-level format to the new 34-province 2-level format (effective 01/07/2025).

## Usage

```python
from underthesea import convert_address

result = convert_address("Phường Phúc Xá, Quận Ba Đình, Thành phố Hà Nội")
print(result.converted)      # "Phường Hồng Hà, Thành phố Hà Nội"
print(result.status)         # ConversionStatus.SUCCESS
print(result.mapping_type)   # MappingType.MERGED
```

## Function Signature

```python
def convert_address(address: str) -> ConversionResult
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `address` | `str` | Vietnamese address string in old format (e.g. `"Phường X, Quận Y, Tỉnh Z"`) |

## Returns

| Type | Description |
|------|-------------|
| `ConversionResult` | Object containing conversion details |

### ConversionResult

| Field | Type | Description |
|-------|------|-------------|
| `original` | `str` | The input address |
| `converted` | `str` | The converted address in new format |
| `status` | `ConversionStatus` | `SUCCESS`, `PARTIAL`, or `NOT_FOUND` |
| `mapping_type` | `MappingType \| None` | `UNCHANGED`, `RENAMED`, `MERGED`, or `DIVIDED` |
| `old` | `AdminUnit` | Parsed old address components |
| `new` | `AdminUnit` | New address components |
| `note` | `str` | Additional information |

### ConversionStatus

| Value | Description |
|-------|-------------|
| `SUCCESS` | Both province and ward matched |
| `PARTIAL` | Only province matched (ward not found) |
| `NOT_FOUND` | Province not recognized |

### MappingType

| Value | Count | Description |
|-------|-------|-------------|
| `UNCHANGED` | 149 | Ward kept the same name |
| `RENAMED` | 92 | Ward renamed (1:1) |
| `MERGED` | 9,328 | Multiple old wards merged into one new ward |
| `DIVIDED` | 1,033 | One old ward split into multiple new wards |

## Examples

### Basic Conversion

```python
from underthesea import convert_address

result = convert_address("Phường Phúc Xá, Quận Ba Đình, Thành phố Hà Nội")
print(result.converted)
# "Phường Hồng Hà, Thành phố Hà Nội"
```

### Abbreviation Support

```python
# P. = Phường, Q. = Quận, TP. = Thành phố
result = convert_address("P. Phúc Xá, Q. Ba Đình, TP. Hà Nội")
print(result.converted)
# "Phường Hồng Hà, Thành phố Hà Nội"
```

### Address with Street

```python
result = convert_address("123 Phố Hàng Bông, Phường Phúc Xá, Quận Ba Đình, TP Hà Nội")
print(result.converted)
# "123 Phố Hàng Bông, Phường Hồng Hà, Thành phố Hà Nội"
```

### Province-only

```python
result = convert_address("Tỉnh Hà Giang")
print(result.status)     # ConversionStatus.PARTIAL
print(result.converted)  # "Tỉnh Tuyên Quang"
```

### Batch Conversion

```python
from underthesea.address import batch_convert

addresses = [
    "Phường Phúc Xá, Quận Ba Đình, Thành phố Hà Nội",
    "Phường Tân Lộc, Quận Thốt Nốt, Thành phố Cần Thơ",
]
results = batch_convert(addresses)
for r in results:
    print(f"{r.original} → {r.converted}")
```

## Supported Abbreviations

| Abbreviation | Expansion |
|---|---|
| `TP.`, `T.P.` | Thành phố |
| `P.` | Phường |
| `Q.` | Quận |
| `H.` | Huyện |
| `TX.`, `T.X.` | Thị xã |
| `TT.`, `T.T.` | Thị trấn |
| `X.` | Xã |

## Notes

- Covers 10,602 ward mapping records across all 63 old provinces
- The district level (quận/huyện) is removed in the new format
- For divided wards, the default new ward is selected
- Uses `underthesea.text_normalize` character normalization for encoding fixes
