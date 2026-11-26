from unittest import TestCase

from underthesea import pos_tag


class TestPosTagRestoreDiacritics(TestCase):
    """pos_tag(..., restore_diacritics=True) for text without diacritics (#200)."""

    def test_issue_200_example(self):
        actual = pos_tag("lap trinh vien tri tue nhan tao", restore_diacritics=True)
        expected = [("lập trình viên", "N"), ("trí tuệ", "N"), ("nhân tạo", "A")]
        self.assertEqual(expected, actual)

    def test_model_v2(self):
        actual = pos_tag("lap trinh vien tri tue nhan tao", model="v2.0", restore_diacritics=True)
        self.assertEqual(["lập trình viên", "trí tuệ", "nhân tạo"], [word for word, _ in actual])

    def test_default_is_unchanged(self):
        words = [word for word, _ in pos_tag("lap trinh vien tri tue nhan tao")]
        self.assertNotIn("lập trình viên", words)

    def test_empty(self):
        self.assertEqual([], pos_tag("", restore_diacritics=True))
