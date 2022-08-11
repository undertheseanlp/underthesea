from unittest import TestCase

from underthesea import text_normalize


class TestTextNormalize(TestCase):
    def test_1(self):
        text = "Ðảm baỏ chất lựơng"
        actual = text_normalize(text)
        expected = "Đảm bảo chất lượng"
        self.assertEqual(actual, expected)
