from os.path import dirname, join
from unittest import TestCase

from underthesea.pipeline.ipa import vietnamese_to_ipa


class TestIPA(TestCase):
    def test_1(self):
        # text = "cún"
        text = "chật"
        actual = vietnamese_to_ipa(text)
        # expected = "kun³⁴"
        expected = "ʨɤ̆t¹⁰ˀ"
        self.assertEqual(expected, actual)

    def test_2(self):
        text = 'hai âm tiết'
        actual = vietnamese_to_ipa(text)
        expected = ""
        self.assertEqual(expected, actual)

    def test_3(self):
        cw = dirname(__file__)
        with open(join(cw, 'tests.txt')) as f:
            lines = f.readlines()
        items = [line.strip() for line in lines]
        for item in items:
            syllable, ipa = item.split(",")
            actual = vietnamese_to_ipa(syllable)
            self.assertEqual(ipa, actual)
