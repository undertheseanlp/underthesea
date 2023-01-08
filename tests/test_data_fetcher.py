from underthesea.data_fetcher import DataFetcher
from unittest import TestCase


class TestDataFetcher(TestCase):
    def test_1(self):
        data = "CP_Vietnamese_VLC_v2_2022"
        DataFetcher.download_data(data, None)
