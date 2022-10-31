import bz2
from os.path import join
from pathlib import Path
from WikiExtractor import extract, options
from multiprocessing import cpu_count
# Download data
from underthesea.file_utils import DATASETS_FOLDER, get_from_cache
from underthesea.utils import logger


def download_wikidump(timestamp):
    DATA_FOLDER = Path(join(DATASETS_FOLDER, "CP_Vietnamese-WIK"))
    # Download data
    print("Download data")
    file = f"https://dumps.wikimedia.org/viwiki/{timestamp}/viwiki-{timestamp}-pages-articles.xml.bz2"
    get_from_cache(file, cache_dir=DATA_FOLDER)
    # Extract data
    print("Extract data")


def decompress_file(filepath):
    logger.info("Extract file")
    zipfile = bz2.BZ2File(filepath)  # open the file
    data = zipfile.read()  # get the decompressed data
    logger.info("Write file")
    newfilepath = filepath[:-4]  # assuming the filepath ends with .bz2
    open(newfilepath, 'wb').write(data)  # write a uncompressed file
    logger.info('Done')


# Extract data
def extract_data(filepath):
    DATA_FOLDER = Path(join(DATASETS_FOLDER, "CP_Vietnamese-WIK"))
    args = options
    args.links = options.keepLinks
    args.sections = True

    args.html = options.toHTML
    args.json = options.write_json
    args.revision = options.print_revision
    args.no_templates = True
    args.quiet = True
    args.lists = True
    args.debug = False
    args.bytes = '100M'
    args.namespaces = ''
    args.ignored_tags = ''
    args.discard_elements = ''
    args.article = ''
    args.compress = ''
    default_process_count = max(1, cpu_count() - 1)
    args.processes = default_process_count
    args.input = join(DATA_FOLDER, "viwiki-20220801-pages-articles.xml")
    args.output = join(DATA_FOLDER, "viwiki")
    extract(args)


if __name__ == '__main__':
    timestamp = "20220801"
    # download_wikidump(timestamp)
    DATA_FOLDER = Path(join(DATASETS_FOLDER, "CP_Vietnamese-WIK"))
    filepath = join(DATA_FOLDER, f"viwiki-{timestamp}-pages-articles.xml.bz2")
    wiki_filepath = join(DATA_FOLDER, f"viwiki-{timestamp}-pages-articles.xml")
    # decompress_file(filepath)
    extract_data(wiki_filepath)
