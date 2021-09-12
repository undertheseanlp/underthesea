import streamlit as st

st.set_page_config(
    page_title="Add Entry Dictionary",
    layout="wide",
)

word = "A"
st.write('# Dictionary')

col1, col2, col3 = st.columns([10, 1, 40])
col1.text_input('Word')


col3.write("#### Sense")
col3.button('edit')

col3.json({
    'foo': 'bar',
    'examples': ['ab', 'b', 'c']
})
