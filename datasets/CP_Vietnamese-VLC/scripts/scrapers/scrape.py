import json
from thuvienphapluat_scraper import ThuvienphapluatScraper

scraper = ThuvienphapluatScraper()

inital_list_page_url = "https://thuvienphapluat.vn/page/tim-van-ban.aspx?keyword=&area=0&type=10&status=0&lan=1&org=0&signer=0&match=True&sort=1&bdate=26/09/1943&edate=27/09/2023"
pages = scraper.scrape_list_page(inital_list_page_url)
metadata = {}
for page in pages:
    print(page["title"])
    print(page["url"])
    print()
    # write content to file in folder data
    with open("data/" + page["title"] + ".txt", "w", encoding="utf-8") as f:
        f.write(scraper.scrape(page["url"]))
    # add metadata to dictionary
    metadata[page["title"]] = page["url"]
    print()
    print("-" * 80)
    print()

# save metadata to file
with open("metadata.text", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)