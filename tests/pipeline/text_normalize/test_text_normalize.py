from unittest import TestCase

from underthesea import text_normalize


class TestTextNormalize(TestCase):
    def test_1(self):
        text = "Ðảm baỏ chất lựơng phòng thí nghịêm hoá học"
        actual = text_normalize(text)
        expected = "Đảm bảo chất lượng phòng thí nghiệm hóa học"
        self.assertEqual(expected, actual)

    def test_special_character(self):
        text = 'a\tb'
        actual = text_normalize(text)
        expected = 'a b'
        self.assertEqual(expected, actual)
