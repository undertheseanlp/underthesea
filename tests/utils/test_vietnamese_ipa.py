from os.path import join, dirname
from unittest import TestCase

from underthesea.utils.vietnamese_ipa import Syllable

DATA_TEST_FOLDER = join(dirname(__file__), "test_data")


def read_test_files(filepath):
    with open(filepath) as f:
        lines = f.read().strip().splitlines()
    return lines


class TestVietnameseIPA(TestCase):
    def test_1(self):
        syllables = ["đây", "đấy", "đó", "đâu"]
        ipas = ["ɗɤ̆j³³", "ɗɤ̆j³⁴", "ɗɔ³⁴", "ɗɤ̆w³³"]
        for i, s in enumerate(syllables):
            syllable = Syllable(s)
            actual = syllable.generate_ipa()
            expected = ipas[i]
            self.assertEqual(expected, actual)

    def test_2(self):
        syllables = ["ao", "au"]
        ipas = ["aw³³", "ăw³³"]
        for i, s in enumerate(syllables):
            syllable = Syllable(s)
            actual = syllable.generate_ipa()
            expected = ipas[i]
            self.assertEqual(expected, actual)

    def test_3(self):
        syllables = ["bao", "bau", "bay", "bai"]
        ipas = ["baw³³", "băw³³", "băj³³", "baj³³"]
        for i, s in enumerate(syllables):
            syllable = Syllable(s)
            actual = syllable.generate_ipa()
            expected = ipas[i]
            self.assertEqual(expected, actual)

    def test_4(self):
        syllables = ["buộc"]
        ipas = ["ɓuək²¹"]
        for i, s in enumerate(syllables):
            syllable = Syllable(s)
            actual = syllable.generate_ipa()
            expected = ipas[i]
            self.assertEqual(expected, actual)

    def test_rimes_n_c_p(self):
        inputs = read_test_files(join(DATA_TEST_FOLDER, "rimes.in"))
        expected = read_test_files(join(DATA_TEST_FOLDER, "rimes_n-c-p.out"))
        for i, item in enumerate(inputs):
            tokens = item.lower().split()
            actual = " ".join([Syllable(token).generate_ipa() for token in tokens])
            self.assertEqual(expected[i], actual)
