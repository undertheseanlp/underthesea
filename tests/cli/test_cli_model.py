# -*- coding: utf-8 -*-
from click.testing import CliRunner
from unittest import TestCase, skip
from underthesea.cli import download_model, list_model


class TestCliModel(TestCase):
    def test_download(self):
        runner = CliRunner()
        result = runner.invoke(download_model, ['SA_GENERAL'])
        print(result.output)

    @skip
    def test_list_model(self):
        runner = CliRunner()
        result = runner.invoke(list_model, ['--all'])
        print(result.output)
