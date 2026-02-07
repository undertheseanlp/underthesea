"""Tests for address converter."""

from underthesea.address import convert_address, batch_convert, ConversionStatus, MappingType


class TestMergedWard:
    """Test wards that were merged into new wards."""

    def test_phuc_xa_merged_to_hong_ha(self):
        result = convert_address("Phường Phúc Xá, Quận Ba Đình, Thành phố Hà Nội")
        assert result.status == ConversionStatus.SUCCESS
        assert result.mapping_type == MappingType.MERGED
        assert result.new.ward == "Phường Hồng Hà"
        assert result.new.province == "Thành phố Hà Nội"

    def test_an_binh_can_tho(self):
        result = convert_address("Phường An Bình, Quận Ninh Kiều, Thành phố Cần Thơ")
        assert result.status == ConversionStatus.SUCCESS
        assert result.new.province == "Thành phố Cần Thơ"


class TestUnchangedWard:
    """Test wards that remain unchanged."""

    def test_tan_loc_can_tho(self):
        result = convert_address("Phường Tân Lộc, Quận Thốt Nốt, Thành phố Cần Thơ")
        assert result.status == ConversionStatus.SUCCESS
        assert result.mapping_type == MappingType.UNCHANGED
        assert result.new.ward == "Phường Tân Lộc"


class TestRenamedWard:
    """Test wards that were renamed."""

    def test_long_hoa_renamed(self):
        result = convert_address("Phường Long Hòa, Quận Bình Thủy, Thành phố Cần Thơ")
        assert result.status == ConversionStatus.SUCCESS
        assert result.mapping_type == MappingType.RENAMED
        assert result.new.ward == "Phường Long Tuyền"


class TestDividedWard:
    """Test wards that were split into multiple new wards."""

    def test_divided_selects_default(self):
        result = convert_address("Xã Tân Thạnh, Huyện Thới Lai, Thành phố Cần Thơ")
        assert result.status == ConversionStatus.SUCCESS
        assert result.mapping_type == MappingType.DIVIDED


class TestAbbreviations:
    """Test address abbreviation expansion."""

    def test_p_q_tp(self):
        result = convert_address("P. Phúc Xá, Q. Ba Đình, TP. Hà Nội")
        assert result.status == ConversionStatus.SUCCESS
        assert result.new.ward == "Phường Hồng Hà"
        assert result.new.province == "Thành phố Hà Nội"

    def test_tp_shorthand(self):
        result = convert_address("P.Phúc Xá, Q.Ba Đình, TP.Hà Nội")
        assert result.status == ConversionStatus.SUCCESS
        assert result.new.province == "Thành phố Hà Nội"


class TestPartialAddress:
    """Test partial addresses (province-only, no ward)."""

    def test_province_only(self):
        result = convert_address("Thành phố Hà Nội")
        assert result.status == ConversionStatus.PARTIAL
        assert result.new.province == "Thành phố Hà Nội"

    def test_unknown_province(self):
        result = convert_address("Tỉnh Không Tồn Tại")
        assert result.status == ConversionStatus.NOT_FOUND


class TestWithStreet:
    """Test addresses that include street information."""

    def test_street_preserved(self):
        result = convert_address("123 Phố Hàng Bông, Phường Phúc Xá, Quận Ba Đình, Thành phố Hà Nội")
        assert result.status == ConversionStatus.SUCCESS
        assert "123" in result.converted
        assert "Phường Hồng Hà" in result.converted


class TestBatchConvert:
    """Test batch conversion."""

    def test_batch(self):
        addresses = [
            "Phường Phúc Xá, Quận Ba Đình, Thành phố Hà Nội",
            "Phường Tân Lộc, Quận Thốt Nốt, Thành phố Cần Thơ",
            "Tỉnh Không Tồn Tại",
        ]
        results = batch_convert(addresses)
        assert len(results) == 3
        assert results[0].status == ConversionStatus.SUCCESS
        assert results[1].status == ConversionStatus.SUCCESS
        assert results[2].status == ConversionStatus.NOT_FOUND


class TestProvinceMapping:
    """Test province-level conversions (63 -> 34)."""

    def test_ha_noi_stays(self):
        result = convert_address("Thành phố Hà Nội")
        assert result.new.province == "Thành phố Hà Nội"

    def test_merged_province(self):
        result = convert_address("Tỉnh Hà Giang")
        assert result.status == ConversionStatus.PARTIAL
        assert result.new.province
