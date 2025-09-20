import requests
from bs4 import BeautifulSoup
from datetime import datetime


url = "https://thuvienphapluat.vn/chinh-sach-phap-luat-moi/vn/thoi-su-phap-luat/tu-van-phap-luat/42248/danh-muc-luat-bo-luat-hien-hanh-tai-viet-nam"

now = datetime.now()
date_string = now.strftime("%Y-%m-%d %H:%M:%S")
print(f"{date_string} Crawl {url}")


response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

parent_element = soup.find(class_="divModelDetail")

links = parent_element.find_all("a")

filepath = "list_urls.txt"
with open(filepath, "w") as f:
    f.write("")

f = open(filepath, "a")
for link in links:
    url = link["href"] + "\n"
    f.write(url)

f.close()
