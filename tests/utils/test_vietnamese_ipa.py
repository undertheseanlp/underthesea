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
        ipas = ["ɗəj³³", "ɗəj²⁴", "ɗɔ²⁴", "ɗəw³³"]
        for i, s in enumerate(syllables):
            syllable = Syllable(s)
            actual = syllable.generate_ipa()
            expected = ipas[i]
            self.assertEqual(expected, actual)

    def test_2(self):
        syllables = ["ao", "au"]
        ipas = ["ʔaːw³³", "ʔaw³³"]
        for i, s in enumerate(syllables):
            syllable = Syllable(s)
            actual = syllable.generate_ipa()
            expected = ipas[i]
            self.assertEqual(expected, actual)

    def test_3(self):
        syllables = ["bao", "bau", "bay", "bai"]
        ipas = ["ɓaːw³³", "ɓaw³³", "ɓaj³³", "ɓaːj³³"]
        for i, s in enumerate(syllables):
            syllable = Syllable(s)
            actual = syllable.generate_ipa()
            expected = ipas[i]
            self.assertEqual(expected, actual)

    def test_4(self):
        syllables = ["khách"]
        ipas = ["xɛk⁴⁵"]
        for i, s in enumerate(syllables):
            syllable = Syllable(s)
            actual = syllable.generate_ipa(eight=True)
            expected = ipas[i]
            self.assertEqual(expected, actual)

    def test_5(self):
        syllables = ["vua"]
        ipas = ["vuə³³"]
        for i, s in enumerate(syllables):
            syllable = Syllable(s)
            actual = syllable.generate_ipa(eight=True)
            expected = ipas[i]
            self.assertEqual(expected, actual)

    def test_edge_n_c_p(self):
        inputs = read_test_files(join(DATA_TEST_FOLDER, "edge.in"))
        expected = read_test_files(join(DATA_TEST_FOLDER, "edge_n-c-p.out"))
        for i, item in enumerate(inputs):
            tokens = item.lower().split()
            actual = " ".join([Syllable(token).generate_ipa(eight=True) for token in tokens])
            self.assertEqual(expected[i], actual)

    def test_onset_n_c_p(self):
        inputs = read_test_files(join(DATA_TEST_FOLDER, "onsets.in"))
        expected = read_test_files(join(DATA_TEST_FOLDER, "onsets_n-c-p.out"))
        for i, item in enumerate(inputs):
            tokens = item.lower().split()
            actual = " ".join([Syllable(token).generate_ipa(eight=True) for token in tokens])
            self.assertEqual(expected[i], actual)

    def test_rimes_n_c_p(self):
        inputs = read_test_files(join(DATA_TEST_FOLDER, "rimes.in"))
        expected = read_test_files(join(DATA_TEST_FOLDER, "rimes_n-c-p.out"))
        for i, item in enumerate(inputs):
            tokens = item.lower().split()
            actual = " ".join([Syllable(token).generate_ipa(eight=True) for token in tokens])
            self.assertEqual(expected[i], actual)

    def test_tones_n_c_p(self):
        inputs = read_test_files(join(DATA_TEST_FOLDER, "tones.in"))
        expected = read_test_files(join(DATA_TEST_FOLDER, "tones_n-c-p.out"))
        for i, item in enumerate(inputs):
            tokens = item.lower().split()
            actual = " ".join([Syllable(token).generate_ipa(eight=True) for token in tokens])
            self.assertEqual(expected[i], actual)

    def test_vowels_n_c_p(self):
        inputs = read_test_files(join(DATA_TEST_FOLDER, "vowels.in"))
        expected = read_test_files(join(DATA_TEST_FOLDER, "vowels_n-c-p.out"))
        for i, item in enumerate(inputs):
            tokens = item.lower().split()
            actual = " ".join([Syllable(token).generate_ipa(eight=True) for token in tokens])
            self.assertEqual(expected[i], actual)
