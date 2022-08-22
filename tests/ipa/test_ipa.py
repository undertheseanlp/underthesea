from unittest import TestCase

from underthesea.pipeline.ipa import vietnamese_to_ipa


class TestIPA(TestCase):
    def test_1(self):
        text = "cún"
        actual = vietnamese_to_ipa(text)
        expected = "kun³⁴"
        self.assertEqual(expected, actual)

    def test_2(self):
        text = 'hai từ'
        actual = vietnamese_to_ipa(text)
        expected = ""
        self.assertEqual(expected, actual)
