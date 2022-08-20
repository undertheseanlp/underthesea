import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Corpus",
    layout="wide",
)

st.write("""
# Vietnamese Wiki Corpus
## Wikidump data August 2021
""")
c1, c2, c3 = st.columns([1, 1, 1])

df = pd.DataFrame([
    {"": "Language", "value": "Vietnamese"},
    {"": "Corpus description", "value": "Vietnamese Wiki Corpus"},
    {"": "Tagset", "value": "10"},
    {"": "Grammar", "value": "0"},
])

c1.write("""
# General Info
""")
c1.dataframe(df.assign(title='').set_index('title'))

df_count = pd.DataFrame([
    {"": "Tokens", "value": "0"},
    {"": "Words", "value": "0"},
    {"": "Sentences", "value": "0"},
    {"": "Documents", "value": "0"},
])
c2.write("""
# Counts
""")
c2.dataframe(df_count.assign(title='').set_index('title'))

c3.write("""
# Lexicon Sizes
""")
df_lexicon = pd.DataFrame([
    {"": "Word", "value": "0"},
    {"": "Lemma", "value": "0"},
    {"": "lc", "value": "0"},
    {"": "lemma_lc", "value": "0"},
])
c3.dataframe(df_lexicon.assign(title='').set_index('title'))

s2_c1, s2_c2 = st.columns([1, 1])
s2_c1.write("""
# Common Tags
""")
