import os
from os.path import join
import streamlit as st
import streamlit.components.v1 as components

from col_data import Dictionary

st.set_page_config(
    page_title="Dictionary App",
    layout="wide",
)
json_viewer_build_dir = join(
    os.path.dirname(os.path.abspath(__file__)),
    "components", "json_viewer", "component", "frontend", "build"
)
json_viewer = components.declare_component(
    "json_viewer",
    path=json_viewer_build_dir
)
dictionary = Dictionary()
init_word = 'a'


def switch_word(word):
    st.session_state['current_word'] = word
    st.session_state['current_word_data'] = dictionary.get_word(word)
    st.session_state['current_next_words'] = dictionary.get_next_words(word)


if __name__ == '__main__':
    if 'current_word' not in st.session_state:
        switch_word(init_word)

    # SIDEBAR
    m = st.markdown("""
    <style>
    div.stButton > button {
        width: 100%;
    }
    </style>""", unsafe_allow_html=True)

    placeholder = st.sidebar.empty()
    search_text_box = placeholder.text_input('Word', value=st.session_state['current_word'], key='sidebar_text_input')
    if search_text_box:
        switch_word(search_text_box)

    buttons = {}
    for word in st.session_state['current_next_words']:
        buttons[word] = st.sidebar.button(label=word, key=word)
        if buttons[word]:
            switch_word(word)

    st.write('# Dictionary')
    data = st.session_state['current_word_data']

    output_data = json_viewer(json_object=data, label=0)

    save_button = st.button('Save')
    if save_button:
        dictionary.save(st.session_state['current_word'], output_data)
