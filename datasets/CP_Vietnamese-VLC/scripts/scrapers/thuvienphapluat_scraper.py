from bs4 import BeautifulSoup
import requests
import re


class ThuvienphapluatScraper:
    """
    A web scraper for Thuvienphapluat website.

    This scraper takes a URL of the page and returns the cleaned textual content of the article.
    The scraped text is cleaned by removing unwanted spaces, tabs, and newlines.

    Attributes:
    None

    Methods:
    scrape(url: str) -> str:
        Scrapes and returns the cleaned textual content of the article from the given url.
    """
    def __init__(self):
        pass

    def scrape(self, url: str) -> str:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        parent_element = soup.find(class_="cldivContentDocVn")

        # Find the child element with the class "content1" within the parent element
        child_element = parent_element.find(class_="content1")

        content = ""
        paragraphs = child_element.find_all("p")
        for p in paragraphs:
            text = p.get_text() + "\n"
            text = text.replace("\r\n", " ")
            text = text.strip()
            content += text + "\n"
        content = content.strip()

        # remove any two or more consecutive newlines
        content = re.sub(r'\n{2,}', '\n', content)

        # remove any two or more consecutive spaces
        content = re.sub(r' {2,}', ' ', content)

        # remove any two or more consecutive tabs
        content = re.sub(r'\t{2,}', '\t', content)
        # remove spaces before and after newlines
        content = re.sub(r'\s*\n\s*', '\n', content)
        return content
