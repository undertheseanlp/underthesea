# -*- coding: utf-8 -*-
from unittest import TestCase

from underthesea.transformer.tagged import TaggedTransformer


class TestFeaturizers(TestCase):
    def test_transform_1(self):
        templates = [
            "T[0]", "T[1]"
        ]
        transformer = TaggedTransformer(templates)
        sentences = [
            [["Messi", "X"], ["giành", "X"], ["quả", "X"]],
            [["bóng", "X"], ["vàng", "X"]]
        ]
        output = transformer.transform(sentences)
        expected = [
            [['T[0]=Messi', 'T[1]=giành'], ['T[0]=giành', 'T[1]=quả'], ['T[0]=quả', 'T[1]=EOS']],
            [['T[0]=bóng', 'T[1]=vàng'], ['T[0]=vàng', 'T[1]=EOS']]
        ]
        self.assertEqual(output, expected)
