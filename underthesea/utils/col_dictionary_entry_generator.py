from os.path import dirname, join
import pandas as pd

PROJECT_FOLDER = dirname(dirname(dirname(__file__)))
DATASETS_FOLDER = join(PROJECT_FOLDER, "datasets")
COL_FOLDER = join(DATASETS_FOLDER, "UD_Vietnamese-COL")
DICTIONARY_FOLDER = join(DATASETS_FOLDER, "UD_Vietnamese-COL", "dictionary")
DICTIONARY_DATA_FOLDER = join(DICTIONARY_FOLDER, "data")

entry_file = join(DICTIONARY_DATA_FOLDER, "entry.xlsx")


writer = pd.ExcelWriter(entry_file, engine='xlsxwriter')

workbook = writer.book
ws = workbook.add_worksheet('Sheet1')
ws.set_column(0, 0, 20)
ws.set_column(1, 1, 20)
ws.write(0, 0, 'abc')
ws.write(0, 0, 'HEADWORD')
ws.write('A3', 'SENSE #1')
ws.write('A4', 'word class')
ws.write('A5', 'examples')

writer.save()
