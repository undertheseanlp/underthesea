from unittest import TestCase

from underthesea import restore_diacritics


class TestRestoreDiacritics(TestCase):
    def test_simple(self):
        actual = restore_diacritics("toi yeu viet nam")
        expected = "tôi yêu việt nam"
        self.assertEqual(expected, actual)

    def test_context(self):
        # "co the" needs context to become "có thể"
        actual = restore_diacritics("chung ta co the lam duoc")
        expected = "chúng ta có thể làm được"
        self.assertEqual(expected, actual)

    def test_capitalization(self):
        actual = restore_diacritics("Chuc mung nam moi")
        expected = "Chúc mừng năm mới"
        self.assertEqual(expected, actual)

    def test_uppercase(self):
        actual = restore_diacritics("VIET NAM vo dich")
        expected = "VIỆT NAM vô địch"
        self.assertEqual(expected, actual)

    def test_punctuation_and_numbers(self):
        actual = restore_diacritics("toi co 2 con meo.")
        expected = "tôi có 2 con mèo."
        self.assertEqual(expected, actual)

    def test_number_as_context(self):
        # numbers act as the <num> context token instead of breaking context
        actual = restore_diacritics("Toi sinh ngay 15 thang 3 nam 1990")
        expected = "Tôi sinh ngày 15 tháng 3 năm 1990"
        self.assertEqual(expected, actual)

    def test_amount(self):
        actual = restore_diacritics("chuyen 10 trieu dong")
        expected = "chuyển 10 triệu đồng"
        self.assertEqual(expected, actual)

    def test_attached_punctuation(self):
        # words joined by punctuation without spaces are still restored
        actual = restore_diacritics("hom nay,toi di hoc")
        expected = "hôm nay,tôi đi học"
        self.assertEqual(expected, actual)

    def test_non_vietnamese_tokens(self):
        # emails, URLs and foreign words must not be modified
        actual = restore_diacritics("email cua toi la test@gmail.com")
        expected = "email của tôi là test@gmail.com"
        self.assertEqual(expected, actual)

    def test_already_accented(self):
        text = "Tôi yêu Việt Nam"
        self.assertEqual(text, restore_diacritics(text))

    def test_whitespace_preserved(self):
        actual = restore_diacritics("toi  yeu\tviet nam")
        expected = "tôi  yêu\tviệt nam"
        self.assertEqual(expected, actual)

    def test_empty(self):
        self.assertEqual("", restore_diacritics(""))

    def test_no_letters(self):
        text = "123 !!! 456"
        self.assertEqual(text, restore_diacritics(text))
