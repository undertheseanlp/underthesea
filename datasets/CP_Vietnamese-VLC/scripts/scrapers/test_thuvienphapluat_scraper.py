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
        # cwd = os.path.dirname(__file__)
        # with open(os.path.join(cwd, "out.txt"), "w") as f:
        #     f.write(result)
        expected = self.read_testdata("1.out")
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
