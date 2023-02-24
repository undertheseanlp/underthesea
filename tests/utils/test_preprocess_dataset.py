# -*- coding: utf-8 -*-
from unittest import TestCase
from datasets import load_dataset
from underthesea.utils.preprocess_dataset import preprocess_word_tokenize_dataset


class TestPreprocessDataset(TestCase):
    def test_preprocess_word_tokenize_dataset(self):
        dataset = load_dataset("undertheseanlp/UTS_WTK_v1")
        new_dataset = preprocess_word_tokenize_dataset(dataset)
        self.assertEqual(len(new_dataset['train']), 8000)
        self.assertEqual(len(new_dataset['validation']), 1000)
        self.assertEqual(len(new_dataset['test']), 1000)
