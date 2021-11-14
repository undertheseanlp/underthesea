# -*- coding: utf-8 -*-
from unittest import TestCase

from underthesea.transformer.tagged import TaggedTransformer
from underthesea_core import featurizer, CRFFeaturizer

from underthesea.transformer.tagged_feature import lower_words


class TestFeaturizers(TestCase):
    def test_transform_1(self):
        templates = [
            "T[0].lower",
            "T[0].isdigit",
            "T[0].istitle",
            "T[0].is_in_dict",

            # word unigram and bigram and trigram
            "T[-1]", "T[0]", "T[1]",
            "T[0,1]",
            "T[0,2]",
        ]
        transformer = TaggedTransformer(templates)
        sentences = [
            [["Messi", "X"], ["giành", "X"], ["quả", "X"], ["bóng", "X"], ["vàng", "X"]]
        ]
        output1 = transformer.transform(sentences)
        expected = [
            [
                ['T[0].lower=messi',
                 'T[0].isdigit=False',
                 'T[0].istitle=True',
                 'T[0].is_in_dict=False',
                 'T[-1]=BOS',
                 'T[0]=Messi',
                 'T[1]=giành',
                 'T[0,1]=Messi giành',
                 'T[0,2]=Messi giành quả'
                 ],
                ['T[0].lower=giành',
                 'T[0].isdigit=False',
                 'T[0].istitle=False',
                 'T[0].is_in_dict=True',
                 'T[-1]=Messi',
                 'T[0]=giành',
                 'T[1]=quả',
                 'T[0,1]=giành quả',
                 'T[0,2]=giành quả bóng'],
                ['T[0].lower=quả',
                 'T[0].isdigit=False',
                 'T[0].istitle=False',
                 'T[0].is_in_dict=True',
                 'T[-1]=giành',
                 'T[0]=quả',
                 'T[1]=bóng',
                 'T[0,1]=quả bóng',
                 'T[0,2]=quả bóng vàng'],
                ['T[0].lower=bóng',
                 'T[0].isdigit=False',
                 'T[0].istitle=False',
                 'T[0].is_in_dict=True',
                 'T[-1]=quả',
                 'T[0]=bóng',
                 'T[1]=vàng',
                 'T[0,1]=bóng vàng',
                 'T[0,2]=EOS'],
                ['T[0].lower=vàng',
                 'T[0].isdigit=False',
                 'T[0].istitle=False',
                 'T[0].is_in_dict=True',
                 'T[-1]=bóng',
                 'T[0]=vàng',
                 'T[1]=EOS',
                 'T[0,1]=EOS',
                 'T[0,2]=EOS']]]

        output2 = featurizer(sentences, templates, lower_words)

        crf_featurizer = CRFFeaturizer(templates, lower_words)
        output3 = crf_featurizer.process(sentences)
        self.assertEqual(output1, output2)
        self.assertEqual(output2, output3)
        self.assertEqual(output1, expected)
