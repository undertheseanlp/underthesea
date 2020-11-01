# -*- coding: utf-8 -*-
from click.testing import CliRunner
from unittest import TestCase
from underthesea.cli import download_model


class TestCliModel(TestCase):
    def test_download(self):
        runner = CliRunner()
        result = runner.invoke(download_model, ['SA_GENERAL'])
        print(result.output)
