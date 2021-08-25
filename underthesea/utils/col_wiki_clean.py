import os
from os import listdir
from os.path import join
from underthesea import sent_tokenize
from multiprocessing import Pool

from underthesea.file_utils import UNDERTHESEA_FOLDER

WIKI_FOLDER = join(UNDERTHESEA_FOLDER, "data", "viwiki-20210720")
RAW_FOLDER = join(WIKI_FOLDER, "text", "AA")
CLEANED_FOLDER = join(WIKI_FOLDER, "cleaned", "AA")

os.makedirs(CLEANED_FOLDER, exist_ok=True)


def check_line(line):
    if len(line) < 30:
        return False
    if line.startswith("<"):
        return False
    return True


def clean(file):
    print(file)
    with open(join(RAW_FOLDER, file)) as f:
        content = f.read()
    with open(join(CLEANED_FOLDER, file), 'w') as out_file:
        out_file.write('')
    with open(join(CLEANED_FOLDER, file), 'a') as out_file:
        for line in content.split("\n"):
            if not check_line(line):
                continue
            sents = sent_tokenize(line)
            for sent in sents:
                if not check_line(sent):
                    continue
                out_file.write(sent.strip() + '\n')
    return


if __name__ == '__main__':
    files = sorted(listdir(RAW_FOLDER))
    with Pool(10) as p:
        p.map(clean, files)
