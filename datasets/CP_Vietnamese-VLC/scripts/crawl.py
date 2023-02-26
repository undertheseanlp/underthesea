import requests
from bs4 import BeautifulSoup
from os.path import dirname, join, abspath
from datetime import datetime


def extract_filename(url):
    last_slash_index = url.rindex("/")
    last_period_index = url.rindex(".")
    last_slash = last_slash_index + 1
    output = url[last_slash:last_period_index]
    return output


def crawl_website(url, filepath):
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{date_string} Crawl {url}")

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    parent_element = soup.find(class_="cldivContentDocVn")

    # Find the child element with the class "content1" within the parent element
    child_element = parent_element.find(class_="content1")

    paragraphs = child_element.find_all("p")

    with open(filepath, "w") as f:
        f.write("")

    f = open(filepath, "a")
    for p in paragraphs:
        text = p.get_text() + "\n"
        text = text.replace("\r\n", " ")
        f.write(text)

    f.close()


CWD = abspath(dirname(__file__))
OUTPUT_FOLDER = join(dirname(CWD), "output")
print(OUTPUT_FOLDER)
with open(join(CWD, "error.log"), "w") as f:
    f.write("")
log_file = open(join(CWD, "error.log"), "a")
with open(join(CWD, "list_urls.txt")) as f:
    urls = f.read().strip().splitlines()
for i, url in enumerate(urls):
    filepath = extract_filename(url)
    try:
        crawl_website(url, join(OUTPUT_FOLDER, f"{filepath}.txt"))
    except Exception as e:
        print(e)
        log_file.write(url + "\n")
