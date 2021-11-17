from unittest import TestCase


class TestTopLevel(TestCase):
    def test_1(self):
        from underthesea import dependency_parse
        print(dependency_parse)
