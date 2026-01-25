import os
from unittest import TestCase, skipUnless


class TestUTSDictionary(TestCase):
    @skipUnless(os.environ.get("RUN_SLOW_TESTS"), "Slow test - requires network")
    def test_load_dictionary(self):
        from underthesea.datasets.uts_dictionary import UTSDictionary

        dictionary = UTSDictionary()
        words = dictionary.words
        self.assertIsInstance(words, list)
        self.assertGreater(len(words), 70000)

    @skipUnless(os.environ.get("RUN_SLOW_TESTS"), "Slow test - requires network")
    def test_contains(self):
        from underthesea.datasets.uts_dictionary import UTSDictionary

        dictionary = UTSDictionary()
        self.assertTrue(len(dictionary) > 0)

    @skipUnless(os.environ.get("RUN_SLOW_TESTS"), "Slow test - requires network")
    def test_get_dictionary(self):
        from underthesea.datasets.uts_dictionary import get_dictionary

        dict1 = get_dictionary()
        dict2 = get_dictionary()
        self.assertIs(dict1, dict2)
