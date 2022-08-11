from unittest import TestCase

from underthesea import text_normalize


class TestTextNormalize(TestCase):
    def test_1(self):
        text = "Ðảm baỏ chất lựơng"
        actual = text_normalize(text)
        expected = "Đảm bảo chất lượng"
        self.assertEqual(actual, expected)

    def test_special_character(self):
        text = 'a\tb'
        actual = text_normalize(text)
        expected = 'a b'
        self.assertEqual(expected, actual)
