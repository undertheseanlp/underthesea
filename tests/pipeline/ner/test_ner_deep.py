# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import ner


class TestNERDeep(TestCase):
    def test_ner_deep_1(self):
        output = ner("Bộ Công Thương xóa một tổng cục, giảm nhiều đầu mối", deep=True)
        for item in output:
            print(item)
        self.assertEqual(output[0]["entity"], "B-ORG")

    def test_ner_deep_2(self):
        output = ner("Chưa tiết lộ lịch trình tới Việt Nam của Tổng thống Mỹ Donald Trump", deep=True)
        # for item in output:
        #     print(item)
        self.assertEqual(output[0]["entity"], "B-LOC")
