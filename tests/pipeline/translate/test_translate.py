# -*- coding: utf-8 -*-
import unittest


class TestTranslate(unittest.TestCase):
    def test_vi_to_en(self):
        from underthesea import translate
        result = translate("Xin chào")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_en_to_vi(self):
        from underthesea import translate
        result = translate("Hello", source_lang='en', target_lang='vi')
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_empty_string(self):
        from underthesea import translate
        result = translate("")
        self.assertEqual(result, "")

    def test_whitespace_only(self):
        from underthesea import translate
        result = translate("   ")
        self.assertEqual(result, "")


if __name__ == '__main__':
    unittest.main()
