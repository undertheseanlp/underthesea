import os
from os.path import join
import requests
from tqdm import tqdm
from underthesea.file_utils import UNDERTHESEA_FOLDER
import subprocess


def download_file(url, dest_file):
    print(f"Download wiki dump from {url}")
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(dest_file, 'wb') as file, tqdm(
            desc=dest_file,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


TS = "20220401"
wiki_folder = join(UNDERTHESEA_FOLDER, "data", f"viwiki-{TS}")

# Create wiki folder
raw_folder = join(wiki_folder, "raw")
if not os.path.exists(raw_folder):
    print(f'Create wiki folder {raw_folder}')
    os.makedirs(raw_folder)

# Download wiki dump
wiki_raw_file = join(raw_folder, f"viwiki-{TS}-pages-articles.xml.bz2")
wiki_extracted_raw_file = join(raw_folder, f"viwiki-{TS}-pages-articles.xml")
url = f"https://dumps.wikimedia.org/viwiki/{TS}/viwiki-{TS}-pages-articles.xml.bz2"
if not os.path.exists(wiki_raw_file) and not wiki_extracted_raw_file:
    download_file(url, wiki_raw_file)

if not wiki_extracted_raw_file:
    # TBD: call bzip2 function
    subprocess.run(["ls", "-l"])

wiki_text_folder = join(wiki_folder, "text")
if not os.path.exists(wiki_text_folder):
    subprocess.run(["python", "WikiExtractor.py", "--no-templates",
                    "-b", "10M",
                    "-s",
                    "--lists",
                    "-o", wiki_text_folder,
                    wiki_extracted_raw_file
                    ])
