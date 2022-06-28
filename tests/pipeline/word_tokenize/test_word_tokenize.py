# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import word_tokenize
import signal


def timeout_handler(signum, frame):
    raise Exception("Timeout Exception")


class TestWordTokenize(TestCase):
    def test_simple_cases(self):
        sentence = u""
        actual = word_tokenize(sentence)
        expected = []
        self.assertEqual(actual, expected)

        actual = word_tokenize(sentence, format="text")
        expected = u""
        self.assertEqual(actual, expected)

    def test_special_cases_2(self):
        sentence = u"="
        actual = word_tokenize(sentence)
        expected = ["="]
        self.assertEqual(actual, expected)

    def test_special_cases_3(self):
        sentence = u"=))"
        actual = word_tokenize(sentence)
        expected = ["=))"]
        self.assertEqual(actual, expected)

    def test_decomposed_from(self):
        text = u"yếu"
        acutal = word_tokenize(text)
        expected = [u'yếu']
        self.assertEqual(acutal, expected)

    def test_word_tokenize(self):
        text = u"""Tổng thống Nga coi việc Mỹ không kích căn cứ quân sự của Syria là "sự gây hấn nhằm vào một quốc gia có chủ quyền", gây tổn hại đến quan hệ Moscow-Washington."""
        word_tokenize(text)

    def test_word_tokenize_2(self):
        """ Case with special character tab
        """
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1)
        try:
            text = u"""000000000000_753889211466429	"""
            word_tokenize(text)
        except Exception as e:
            raise (e)

    def test_url_1(self):
        text = u"https://www.facebook.com/photo.php?fbid=1627680357512432&set=a.1406713109609159.1073741826.100008114498358&type=1 mình muốn chia sẻ bài viết của một bác nói về thực trạng của bộ giáo dục bây giờ! mọi người vào đọc và chia sẻ để Phạm Vũ Luận BIẾT!"
        actual = word_tokenize(text, format='text')
        expected = u"https://www.facebook.com/photo.php?fbid=1627680357512432&set=a.1406713109609159.1073741826.100008114498358&type=1 mình muốn chia_sẻ bài viết của một bác nói về thực_trạng của bộ giáo_dục bây_giờ ! mọi người vào đọc và chia_sẻ để Phạm_Vũ_Luận BIẾT !"
        self.assertEqual(actual, expected)

    # from issue 528
    # link: https://github.com/undertheseanlp/underthesea/issues/528
    def test_exception(self):
        text = "000 85 ."
        actual = word_tokenize(text, format='text')
        expected = "000 85 ."
        self.assertEqual(actual, expected)
