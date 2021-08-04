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
