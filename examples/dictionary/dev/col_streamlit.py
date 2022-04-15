import os
from os.path import dirname, join
import streamlit as st
from underthesea.utils.col_dictionary import Dictionary
from underthesea.file_utils import UNDERTHESEA_FOLDER
from underthesea.utils.col_script import UDDataset

PROJECT_FOLDER = dirname(dirname(__file__))
os.sys.path.append(PROJECT_FOLDER)

DICTIONARY_FILE = join(PROJECT_FOLDER, "datasets", "dictionary", "202108.yaml")
dictionary = Dictionary.load(dictionary_file=DICTIONARY_FILE)
dictionary_n_words = len(dictionary.data)
st.set_page_config(
    page_title="Open Dictionary",
    # layout="wide",
)

st.write("""
# Open Dictionary
""")
file = "wiki_00"
ud_file = join(UNDERTHESEA_FOLDER, "data", "viwiki-20210720", "ud", "AA", file)
ud_dataset = UDDataset.load(ud_file)


def find_word(ud_dataset, word) -> list:
    max_samples = 10
    i = 0
    sentences = []
    for s in ud_dataset:
        text_original = s.headers['text']
        text = text_original.lower()
        if word in text:
            sentences.append(text_original)
            i += 1
        if i > max_samples:
            break
    return sentences


st.write('Loaded corpus: wiki')
st.write(f'Loaded dictionary: {dictionary_n_words} words (202108.yaml)')

word = st.text_input('Word')

if word:
    sentences = find_word(ud_dataset, word)
    st.write(f"## {word}")
    st.write("Output:")
    for s in sentences:
        st.write(s)
