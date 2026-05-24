# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import dependency_parse


class TestDependencyParse(TestCase):
    def test_1(self):
        text = 'Tối 29/11, Việt Nam thêm 2 ca mắc Covid-19'
        actual = dependency_parse(text)
        expected = [('Tối', 5, 'obl:tmod'),
                    ('29/11', 1, 'compound'),
                    (',', 4, 'punct'),
                    ('Việt Nam', 5, 'nsubj'),
                    ('thêm', 0, 'root'),
                    ('2', 7, 'nummod'),
                    ('ca', 5, 'obj'),
                    ('mắc', 5, 'obj'),
                    ('Covid-19', 5, 'punct')]
        self.assertEqual(expected, actual)

    def test_batch(self):
        texts = [
            'Tối 29/11, Việt Nam thêm 2 ca mắc Covid-19',
            'Tôi đi học',
        ]
        actual = dependency_parse(texts)
        self.assertEqual(len(actual), 2)
        # batch result for the first text matches the single-text result
        self.assertEqual(actual[0], dependency_parse(texts[0]))
        # each parsed token is a (word, head, relation) triple
        for sentence in actual:
            for token in sentence:
                self.assertEqual(len(token), 3)

    def test_batch_size_does_not_change_result(self):
        # The torch-level batch_size only controls how many tokens share a
        # forward pass; predictions must be identical regardless of its value.
        texts = [
            'Tối 29/11, Việt Nam thêm 2 ca mắc Covid-19',
            'Tôi đi học',
        ]
        one_per_batch = dependency_parse(texts, batch_size=1)
        large_batch = dependency_parse(texts, batch_size=5000)
        self.assertEqual(one_per_batch, large_batch)
