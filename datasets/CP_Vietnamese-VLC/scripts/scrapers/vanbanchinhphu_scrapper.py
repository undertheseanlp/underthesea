import json
import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s [%(levelname)s] - %(message)s', 
                    filename='scraper.log', 
                    filemode='w')

class VanbanchinhphuScraper:
    def __init__(self, client):
        self.base_url = "https://vanban.chinhphu.vn/?pageid=27160&docid="
        self.client = client
        self.db = client.admin  # Use the 'admin' database
        self.jobs_col = self.db.jobs  # 'jobs' collection
        self.docs_col = self.db.docs  # 'docs' collection

    def scrape(self, doc_id: int) -> None:
        job = self.jobs_col.find_one({"_id": doc_id})
        if job and job.get('status') == 'completed':
            logging.info(f"Skipping already completed doc_id {doc_id}")
            return

        logging.info(f"Scraping doc_id {doc_id}")
        url = f"{self.base_url}{doc_id}"
        try:
            response = requests.get(url)
            if response.status_code != 200:
                logging.info(f"Failed to fetch document {doc_id}")
                self.jobs_col.update_one({"_id": doc_id}, {"$set": {"status": "fail"}}, upsert=True)
                return
            
            soup = BeautifulSoup(response.content, "html.parser")
            div = soup.find('div', {'id': 'ctrl_190596_91_Content', 'class': 'Content'})
            
            extracted_data = {}
            if not div:
                self.jobs_col.update_one({"_id": doc_id}, {"$set": {"status": "completed", "has_doc": False}}, upsert=True)
                return
            
            table = div.find('table')
            if not table:
                self.jobs_col.update_one({"_id": doc_id}, {"$set": {"status": "completed", "has_doc": False}}, upsert=True)
                return
            
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) == 2:
                    key = cols[0].get_text(strip=True)
                    link = cols[1].find('a')
                    value = link.get_text(strip=True) if link else cols[1].get_text(strip=True)
                    extracted_data[key] = value
                    
            extracted_data['url'] = url
            self.docs_col.insert_one(extracted_data)
            self.jobs_col.update_one({"_id": doc_id}, {"$set": {"status": "completed", "has_doc": True}}, upsert=True)
            logging.info(f"Done scraping doc_id {doc_id}")

        except Exception as e:
            logging.info(f"Error scraping doc_id {doc_id}: {str(e)}")

            
def worker(doc_ids: range, client) -> None:
    scraper = VanbanchinhphuScraper(client)
    for doc_id in doc_ids:
        scraper.scrape(doc_id)

        
if __name__ == "__main__":
    client = MongoClient('mongodb://root:example@localhost:27017/')
    num_workers = 10
    max_doc_id = 208750

    # Dividing the task between workers
    ids_per_worker = max_doc_id // num_workers
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker, range(i * ids_per_worker + 1, (i + 1) * ids_per_worker + 1), client) for i in range(num_workers)]