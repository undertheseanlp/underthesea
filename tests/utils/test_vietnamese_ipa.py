from unittest import TestCase

from underthesea.utils.vietnamese_ipa import Syllable


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
