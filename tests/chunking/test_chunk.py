# -*- coding: utf-8 -*-
from os import listdir
from underthesea.util.file_io import read, write
from unittest import TestCase, skip
from underthesea import chunk
from os.path import dirname, join

samples_dir = join(dirname(__file__), "samples")


def load_input(input_file):
    lines = read(input_file).strip().split("\n")
    content = [line.split("\t")[0] for line in lines]
    content = u" ".join(content)
    return content


def load_output(input_file):
    lines = [text.split("\t") for text in read(input_file).strip().split("\n")]
    output = [tuple(item) for item in lines]
    return output


def save_temp(id, output):
    temp_file = join(samples_dir, "%s.actual" % id)
    content = u"\n".join([u"\t".join(item) for item in output])
    write(temp_file, content)


class TestChunk(TestCase):
    def test_simple_cases(self):
        sentence = u""
        actual = chunk(sentence)
        expected = []
        self.assertEqual(actual, expected)

    def test_accuracy(self):
        output = chunk(
            u"Tổng Bí thư: Ai trót để tay nhúng chàm thì hãy sớm tự gột rửa")
        self.assertEqual(len(output), 14)
