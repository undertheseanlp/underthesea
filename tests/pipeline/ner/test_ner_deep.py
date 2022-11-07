# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import ner


class TestNERDeep(TestCase):
    def test_ner_deep_1(self):
        output = ner(u"Bộ Công Thương xóa một tổng cục, giảm nhiều đầu mối", deep=True)
        self.assertEqual(output[0]["entity"], "B-ORG")
