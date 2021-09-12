import os
from os.path import join

import streamlit as st
import streamlit.components.v1 as components

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

# SIDEBAR
add_selectbox = st.sidebar.text_input('Word', value="gà")

# MAIN
st.write('# Dictionary')
st.text_input('Word', key=1, value="gà")
data = [
    {
        "description": "(Động vật học) Loài chim nuôi (gia cầm) để lấy thịt và trứng, bay kém, mỏ cứng, con trống có cựa và biết gáy.",
        "tag": "noun",
        "examples": [
            "Bán gà ngày gió, bán chó ngày mưa. (tục ngữ)",
            "Gà người gáy, gà nhà ta sáng. (tục ngữ)"
        ]
    },
    {
        "description": "Đánh cuộc riêng trong một ván bài tổ tôm hay tài bàn ngoài số tiền góp chính",
        "tag": "verb",
        "examples": [
            "Gà lần nào cũng thua thì đánh làm gì."
        ]
    }
]

output_data = json_viewer(json_object=data, label=0)
