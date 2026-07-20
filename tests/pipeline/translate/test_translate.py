import importlib.util
import subprocess
import sys
import unittest


class TestTranslate(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("transformers"),
        "requires the deep-learning dependencies",
    )
    def test_package_import_does_not_load_deep_learning_dependencies(self):
        script = """
import sys
from underthesea import sent_tokenize

loaded = {name for name in ("torch", "transformers") if name in sys.modules}
if loaded:
    raise AssertionError(f"eagerly loaded deep-learning dependencies: {sorted(loaded)}")
"""
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )

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
