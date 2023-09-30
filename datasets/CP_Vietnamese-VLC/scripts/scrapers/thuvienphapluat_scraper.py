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
    

    def scrape_article_page(self, url: str) -> str:
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
    

    def scrape_directory_page(self, url: str) -> dict:
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code != 200:
            response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        article_pages = []
        directory_urls = []
        
        # Find all the <a> tags with the 'onclick' attribute set to 'Doc_CT(MemberGA)'
        for a_tag in soup.find_all('a', onclick='Doc_CT(MemberGA)'):
            title = a_tag.get_text(strip=True)  # Get the text content of the <a> tag (the title of the article)
            url = a_tag.get('href')  # Get the href attribute of the <a> tag (the URL of the article)
            
            # Append the title and URL to the list in the requested structure
            article_pages.append({'title': title, 'url': url})
        

        return article_pages, directory_urls
