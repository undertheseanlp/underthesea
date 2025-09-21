from unittest import TestCase

from process_viwiktionary import Word, split_vietnamese_syllable


class TestModel(TestCase):
    def test_1(self):
        word = Word('chó')
        self.assertTrue(word.is_vietnamese)

    def test_split_vietnamese_syllable(self):
        s = "cún"
        result = split_vietnamese_syllable(s)
        self.assertEqual('c', result[0])

    def test_2(self):
        word = Word('chiến')
        self.assertTrue(word.is_vietnamese)
