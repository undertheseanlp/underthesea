##################################################################
# PREPROCESS ORIGINAL CORPUS: VLSP2020-DP
##################################################################

##################################################################
# CONTENT
##################################################################
# d    4.0K    Jan 16 23:12    DataTestDP2020
# d    4.0K    Jan 16 23:12    DataTestDP2020-CoNLLU
# d    4.0K    Jan 16 23:12    DataTestGoldDP2020
# -    2.9M    Nov 18 09:44    DP-Package2.18.11.2020.txt
# -    665K    Sep 18 09:22    HTB_1570.txt
# -     337    Nov  8 23:09    MotaDulieuDot1.txt
# -     26K    Nov  8 23:07    SA-Hotel-50.txt
# -     20K    Nov  8 23:07    SA-Restaurant-50.txt
# -    1.6M    May 12  2020    VTB_2996.txt
# -    375K    Nov  8 23:07    VTB_400.txt
import os
import shutil
from os.path import join

from underthesea.file_utils import DATASETS_FOLDER

ORIGINAL_ZIP_FILE = join(DATASETS_FOLDER, 'VLSP2020-DP.zip')
ORIGINAL_FOLDER = join(DATASETS_FOLDER, 'VLSP2020-DP-ORIGINAL')
DEST_FOLDER = join(DATASETS_FOLDER, 'VLSP2020-DP')
shutil.rmtree(DEST_FOLDER)
os.makedirs(DEST_FOLDER)


# shutil.rmtree(ORIGINAL_FOLDER)
# with ZipFile(ORIGINAL_ZIP_FILE) as file:
#     file.extractall(ORIGINAL_FOLDER)

class CONLLReader:
    @staticmethod
    def load(file):
        sentences = open(file).read().split('\n\n')
        sentences = [s.strip() for s in sentences if len(s) > 10]
        return sentences

    @staticmethod
    def save(sentences, file):
        n = len(sentences)
        content = '\n\n'.join(sentences) + '\n'
        print(f'Save {n} sentences in {file}')
        with open(file, 'w') as f:
            f.write(content)


files = ['DP-Package2.18.11.2020.txt', 'VTB_2996.txt', 'VTB_400.txt', 'SA-Restaurant-50.txt', 'SA-Hotel-50.txt',
         'HTB_1570.txt']
files = [join(DATASETS_FOLDER, ORIGINAL_FOLDER, f) for f in files]
sentences = []
for file in files:
    sentences += CONLLReader.load(file)
CONLLReader.save(sentences, join(DEST_FOLDER, 'train.txt'))

file = join(ORIGINAL_FOLDER, 'DataTestGoldDP2020', 'total-gold.txt')
sentences = CONLLReader.load(file)
CONLLReader.save(sentences, join(DEST_FOLDER, 'test.txt'))
