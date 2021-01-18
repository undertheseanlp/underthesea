# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import dependency_parse


class TestDependencyParse(TestCase):
    def test_1(self):
        text = 'Tối 29/11, Việt Nam thêm 2 ca mắc Covid-19'
        actual = dependency_parse(text)
        expected = [
            ('Tối', 5, 'obl:tmod'),
            ('29/11', 1, 'flat:time'),
            (',', 5, 'punct'),
            ('Việt Nam', 5, 'nsubj'),
            ('thêm', 0, 'root'),
            ('2', 7, 'nummod'),
            ('ca', 5, 'obj'),
            ('mắc', 7, 'acl:subj'),
            ('Covid-19', 8, 'obj')]
        self.assertEqual(expected, actual)
