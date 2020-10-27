# -*- coding: utf-8 -*-
from click.testing import CliRunner
from unittest import TestCase
from underthesea.cli import download


class TestCliModel(TestCase):
    def test_download(self):
        runner = CliRunner()
        result = runner.invoke(download, ['SA_GENERAL'])
        print(result.output)
