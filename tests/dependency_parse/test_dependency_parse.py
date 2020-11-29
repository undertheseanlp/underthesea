# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import dependency_parse


class TestDependencyParse(TestCase):
    def test_1(self):
        text = 'Tối 29/11, Việt Nam thêm 2 ca mắc Covid-19'
        actual = dependency_parse(text)
        expected = [
            ('Tối', 8, 'punct'),
            ('29/11', 8, 'punct'),
            (',', 8, 'punct'),
            ('Việt Nam', 8, 'punct'),
            ('thêm', 8, 'punct'),
            ('2', 8, 'punct'),
            ('ca', 8, 'punct'),
            ('mắc', 8, 'punct'),
            ('Covid-19', 8, 'punct')
        ]
        self.assertEqual(actual, expected)
