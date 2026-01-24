from unittest import TestCase

from underthesea.pipeline.sentiment import sentiment


class TestSentiment(TestCase):
    def test_bank_labels_returns_list(self):
        self.assertGreater(len(sentiment.bank.labels), 0)

    def test_bank_labels_contains_expected_labels(self):
        expected_labels = ['CARD#negative', 'CARD#positive', 'MONEY_TRANSFER#negative']
        for label in expected_labels:
            self.assertIn(label, sentiment.bank.labels)

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
        text = "nhân viên hỗ trợ quá lâu"
        actual = sentiment(text, domain="bank")
        expected = ['CUSTOMER_SUPPORT#negative']
        self.assertEqual(expected, actual)

    def test_one_label_3(self):
        text = "Dkm t chuyển vẫn bị mất phí"
        actual = [str(label) for label in sentiment(text, domain="bank")]
        expected = ['MONEY_TRANSFER#negative']
        self.assertEqual(sorted(expected), sorted(actual))

    def test_one_label_4(self):
        text = '''TUI cũng bó tay với BIDV Cần Thơ.
                Cả quận NK mà chỉ được lèo tèo mấy thùng ATM và luôn trong
                tình trạng nhìn thấy chữ Sorry cũng nh.ư hết tiền.
                Chán ko buồn nói. Qd có khác '''
        actual = [str(label) for label in sentiment(text, domain="bank")]
        expected = ['CARD#negative']
        self.assertEqual(sorted(expected), sorted(actual))

    def test_one_label_5(self):
        text = 'Có làm thẻ ngân hàng BIDV miễn phí ko'
        actual = sentiment(text, domain="bank")
        expected = ['CARD#negative']
        self.assertEqual(expected, actual)
