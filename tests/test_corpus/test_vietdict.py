from unittest import TestCase
from underthesea.corpus import viet_dict_11K
import sys

if sys.version_info >= (3, 0):
    unicode = str


class Test_VietDict(TestCase):
    def test_viet_dict(self):
        words = viet_dict_11K.words
        self.assertEqual(11373, len(words))

    def test_unicode(self):
        word = viet_dict_11K.words[0]
        self.assertEqual(unicode, type(word))
