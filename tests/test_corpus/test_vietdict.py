from unittest import TestCase, skip
from underthesea.corpus import viet_dict_11K, viet_dict_22K, viet_dict_39K, viet_dict_74K


class Test_VietDict(TestCase):
    def test_viet_dict(self):
        words = viet_dict_11K.words
        self.assertEqual(11373, len(words))

    def test_unicode(self):
        word = viet_dict_11K.words[0]
        self.assertEqual(unicode, type(word))

    @skip("")
    def test_viet_dict_22K(self):
        words = viet_dict_22K.words
        self.assertEqual(22426, len(words))

    @skip("")
    def test_viet_dict_39K(self):
        words = viet_dict_39K.words
        self.assertEqual(39071, len(words))

    @skip("")
    def test_viet_dict_74K(self):
        words = viet_dict_74K.words
        self.assertEqual(73901, len(words))
