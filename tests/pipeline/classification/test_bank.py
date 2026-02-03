from unittest import TestCase

from underthesea import classify


class TestBankClassify(TestCase):
    def test_bank_labels_returns_list(self):
        self.assertGreater(len(classify.bank.labels), 0)

    def test_bank_labels_contains_expected_labels(self):
        expected_labels = ['CARD', 'INTEREST_RATE', 'DISCOUNT']
        for label in expected_labels:
            self.assertIn(label, classify.bank.labels)

    def test_classify_result_in_bank_labels(self):
        text = "Mở tài khoản ATM thì có đc quà ko ad"
        result = classify(text, domain='bank')
        self.assertIn(result[0], classify.bank.labels)

    def test_classify_simple_case(self):
        text = "Mở tài khoản ATM thì có đc quà ko ad"
        actual = classify(text, domain="bank")
        # Convert numpy strings to regular strings for comparison
        actual = [str(label) for label in actual]
        expected = ["CUSTOMER_SUPPORT"]
        self.assertEqual(expected, actual)

    def test_classify_simple_case_2(self):
        text = "Dkm t chuyển vẫn bị mất phí"
        actual = classify(text, domain="bank")
        # Convert numpy strings to regular strings for comparison
        actual = [str(label) for label in actual]
        expected = ["MONEY_TRANSFER"]
        self.assertEqual(expected, actual)

    def test_classify_simple_case_3(self):
        text = "Lãi suất từ BIDV rất ưu đãi"
        actual = classify(text, domain="bank")
        # Convert numpy strings to regular strings for comparison
        actual = [str(label) for label in actual]
        expected = ["INTEREST_RATE"]
        self.assertEqual(expected, actual)
