import unittest
from thuvienphapluat_scraper import ThuvienphapluatScraper
import os


class TestThuvienphapluatScraper(unittest.TestCase):

    def setUp(self):
        self.scrapper = ThuvienphapluatScraper()

    def read_testdata(self, filename):
        cwd = os.path.dirname(__file__)
        filename = os.path.join(cwd, "test_data", filename)
        with open(filename) as f:
            return f.read().strip()

    def test_scrape_1(self):
        url = self.read_testdata("1.in")
        result = self.scrapper.scrape(url)
        expected = self.read_testdata("1.out")
        self.assertEqual(result, expected)

    def test_scrape_list_page_1(self):
        url = "https://thuvienphapluat.vn/page/tim-van-ban.aspx?keyword=&area=0&type=10&status=0&lan=1&org=0&signer=0&match=True&sort=1&bdate=26/09/1943&edate=27/09/2023"
        result = self.scrapper.scrape_list_page(url)
        self.assertTrue(len(result) > 10)
        first_item = result[0]
        self.assertTrue("title" in first_item)
        self.assertTrue("url" in first_item)


if __name__ == '__main__':
    unittest.main()
