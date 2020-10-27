# -*- coding: utf-8 -*-
from click.testing import CliRunner
from unittest import TestCase, skip
from underthesea.cli import download_data, list_data, remove_data


class TestCliData(TestCase):
    @skip
    def test_download_data_raw_file(self):
        runner = CliRunner()
        # data_name = "VNTQ_BIG"
        # data_name = "VNTQ_SMALL"
        data_name = "VNESES"
        result = runner.invoke(download_data, [data_name])
        print(result.output)

    @skip
    def test_download_data_zip_file(self):
        runner = CliRunner()
        data_name = "UTS2017-BANK"
        # url = ""
        result = runner.invoke(download_data, [data_name])
        print(result.output)

    @skip
    def test_list_data(self):
        runner = CliRunner()
        result = runner.invoke(list_data)
        print(result.output)

    @skip
    def test_list_data_2(self):
        runner = CliRunner()
        result = runner.invoke(list_data, ['--all'])
        print(result.output)

    @skip
    def test_remove_data(self):
        runner = CliRunner()
        result = runner.invoke(remove_data, ['VNESES'])
        print(result.output)
