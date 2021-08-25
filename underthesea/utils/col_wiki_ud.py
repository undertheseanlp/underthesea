import os
from os import listdir
from os.path import join

from underthesea.file_utils import UNDERTHESEA_FOLDER
from underthesea.utils import logger
from underthesea.utils.col_script import UDDataset, UDSentence

WIKI_FOLDER = join(UNDERTHESEA_FOLDER, "data", "viwiki-20210720")
CLEANED_FOLDER = join(WIKI_FOLDER, "cleaned", "AA")
UD_FOLDER = join(WIKI_FOLDER, "ud", "AA")
os.makedirs(UD_FOLDER, exist_ok=True)


def check_line(line):
    if len(line) < 30:
        return False
    if line.startswith("<"):
        return False
    return True


def make_ud_file(file):
    logger.info(msg=file)
    sentences = []
    i = 0
    for line in open(join(CLEANED_FOLDER, file)):
        s = UDSentence.load_from_raw_text(line)
        sentences.append(s)
        i += 1
        if i % 200 == 0:
            logger.info(f"{file}:{i}")
    ud_dataset = UDDataset(sentences)
    ud_dataset.write(join(UD_FOLDER, file))
    logger.info(f"Finish {file}")


if __name__ == '__main__':
    files = sorted(listdir(CLEANED_FOLDER))

    # with Pool(5) as p:
    #     p.map(make_ud_file, files)

    # file = files[0]
    # make_ud_file(file)

    for file in files:
        make_ud_file(file)
