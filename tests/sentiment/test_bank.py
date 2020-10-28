# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea.sentiment import sentiment


class TestSentiment(TestCase):
    def test_no_text(self):
        text = ""
        actual = sentiment(text)
        expected = None
        self.assertEqual(actual, expected)

    def test_one_label_1(self):
        text = "Xem lại vẫn thấy xúc động và tự hào về BIDV của mình!"
        actual = [str(label) for label in sentiment(text, domain="bank")]
        expected = ['TRADEMARK#positive']
        self.assertEqual(actual, expected)

    def test_one_label_2(self):
        text = "Đky qua đường link ở bài viết này từ thứ 6 mà giờ chưa thấy ai lhe hết"
        actual = [str(label) for label in sentiment(text, domain="bank")]
        expected = ['CUSTOMER_SUPPORT#negative']
        self.assertEqual(actual, expected)

    def test_multi_label_1(self):
        text = "Dkm t chuyển vẫn bị mất phí"
        actual = [str(label) for label in sentiment(text, domain="bank")]
        expected = ['INTEREST_RATE#negative', 'MONEY_TRANSFER#negative']
        self.assertEqual(sorted(actual), sorted(expected))

    def test_multi_label_2(self):
        text = '''TUI cũng bó tay với BIDV Cần Thơ.
                Cả quận NK mà chỉ được lèo tèo mấy thùng ATM và luôn trong tình trạng nhìn thấy chữ Sorry cũng nh.ư hết tiền.
                Chán ko buồn nói. Qd có khác '''
        actual = [str(label) for label in sentiment(text, domain="bank")]
        expected = ['CARD#negative', 'CUSTOMER_SUPPORT#negative']
        self.assertEqual(sorted(actual), sorted(expected))

    def test_none_label(self):
        text = 'Có làm thẻ ngân hàng BIDV miễn phí ko'
        actual = sentiment(text, domain="bank")
        expected = None
        self.assertEqual(expected, actual)
