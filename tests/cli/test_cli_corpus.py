from click.testing import CliRunner
from unittest import TestCase
from underthesea.cli import validate, revise


class TestCliCorpus(TestCase):
    def test_validate(self):
        runner = CliRunner()
        result = runner.invoke(validate, ['--type=TOKENIZE', '--corpus=VLSP2013-WTK-R1'])
        print(result.output)
        actual = result.exit_code
        expected = 0
        self.assertEqual(expected, actual)

    def test_validate_fail_1(self):
        runner = CliRunner()
        result = runner.invoke(validate, ['--type=UNSUPPORTED_TYPE', '--corpus=VLSP2013-WTK-R1'])
        actual = result.exit_code
        expected = 1
        self.assertEqual(expected, actual)

    def test_validate_fail_2(self):
        runner = CliRunner()
        result = runner.invoke(validate, ['--type=TOKENIZE', '--corpus=DONT-EXIST-DATASET'])
        actual = result.exit_code
        expected = 1
        self.assertEqual(expected, actual)

    def test_revise_corpus(self):
        runner = CliRunner()
        result = runner.invoke(revise, ['--corpus=VLSP2013-WTK'])
        print(result.output)
        actual = result.exit_code
        expected = 0
        self.assertEqual(expected, actual)
