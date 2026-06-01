# -*- coding: utf-8 -*-
from unittest import TestCase
from unittest.mock import patch
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
        # one result per input sentence
        self.assertEqual(len(actual), len(texts))
        # batched results match single-sentence calls, in input order
        self.assertEqual(actual[0], dependency_parse(texts[0]))
        self.assertEqual(actual[1], dependency_parse(texts[1]))

    def test_batch_params(self):
        texts = ['Tôi đi học', 'Hà Nội là thủ đô của Việt Nam']
        actual = dependency_parse(texts, batch_size=2000, buckets=2)
        self.assertEqual(len(actual), len(texts))

    def test_batch_empty(self):
        self.assertEqual(dependency_parse([]), [])

    def test_batch_real_torch_batching(self):
        # Equal-length sentences land in one length bucket and are run through a
        # single torch forward pass (real batching, not a Python loop). This also
        # guards a regression: batches larger than num_layers*2 used to crash.
        from underthesea.pipeline.dependency_parse import init_parser
        parser = init_parser()
        texts = ['Tôi đi học'] * 16
        with patch.object(parser, 'forward', wraps=parser.forward) as spy:
            actual = dependency_parse(texts)
        self.assertEqual(len(actual), 16)
        self.assertEqual(spy.call_count, 1)
