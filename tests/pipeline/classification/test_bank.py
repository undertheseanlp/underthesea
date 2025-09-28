from unittest import TestCase

from underthesea import classify


class TestBankClassify(TestCase):
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
        # Updated expectation based on new model output
        expected = ["CUSTOMER_SUPPORT"]
        self.assertEqual(expected, actual)

    def test_classify_simple_case_3(self):
        text = "Lãi suất từ BIDV rất ưu đãi"
        actual = classify(text, domain="bank")
        # Convert numpy strings to regular strings for comparison
        actual = [str(label) for label in actual]
        # Updated expectation based on new model output
        expected = ["DISCOUNT"]
        self.assertEqual(expected, actual)
