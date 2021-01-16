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
import shutil
from os.path import join
from zipfile import ZipFile

from underthesea.file_utils import DATASETS_FOLDER
ORIGINAL_ZIP_FILE = join(DATASETS_FOLDER, 'VLSP2020-DP.zip')
ORIGINAL_FOLDER = join(DATASETS_FOLDER, 'VLSP2020-DP-ORIGINAL')
shutil.rmtree(ORIGINAL_FOLDER)
with ZipFile(ORIGINAL_ZIP_FILE) as file:
    file.extractall(ORIGINAL_FOLDER)
