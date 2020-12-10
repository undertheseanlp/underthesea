# -*- coding: utf-8 -*-
from unittest import TestCase

from underthesea.utils import generate_table


class TestGenerateTable(TestCase):
    def test_generate_table(self):
        data = [
            (1, 'hihi', 'haha'),
            (2, 'hihi', 'haha'),
        ]
        headers = ['id', 'task', 'score']

        actual = generate_table(data, headers)
        expected = ("| id | task | score |\n"
                    "|----+------+-------|\n"
                    "| 1  | hihi | haha  |\n"
                    "| 2  | hihi | haha  |\n")
        self.assertEqual(actual, expected)
