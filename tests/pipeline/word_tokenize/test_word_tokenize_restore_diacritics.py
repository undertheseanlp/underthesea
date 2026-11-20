from unittest import TestCase

from underthesea import word_tokenize


class TestWordTokenizeRestoreDiacritics(TestCase):
    """word_tokenize(..., restore_diacritics=True) for text without diacritics (#200)."""

    def test_issue_200_example(self):
        actual = word_tokenize("lap trinh vien tri tue nhan tao ha noi", restore_diacritics=True)
        expected = ["lập trình viên", "trí tuệ", "nhân tạo", "hà nội"]
        self.assertEqual(expected, actual)

    def test_text_format(self):
        actual = word_tokenize("chung ta co the lam duoc", format="text", restore_diacritics=True)
        expected = "chúng_ta có_thể làm được"
        self.assertEqual(expected, actual)

    def test_default_is_unchanged(self):
        # without the flag, unaccented input is segmented as before
        sentence = "lap trinh vien tri tue nhan tao"
        actual = word_tokenize(sentence)
        self.assertEqual(sentence, " ".join(actual))
        self.assertNotIn("lập trình viên", actual)

    def test_accented_input_is_kept(self):
        sentence = "Bác sĩ bây giờ có thể thản nhiên báo tin bệnh nhân bị ung thư"
        self.assertEqual(word_tokenize(sentence), word_tokenize(sentence, restore_diacritics=True))

    def test_empty(self):
        self.assertEqual([], word_tokenize("", restore_diacritics=True))
