import os

import streamlit.components.v1 as components
import streamlit as st

_RELEASE = True

if _RELEASE:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(root_dir, "frontend/build")
    _json_viewer = components.declare_component(
        "json_viewer",
        path=build_dir
    )
else:
    _json_viewer = components.declare_component(
        "json_viewer",
        url="http://localhost:3001"
    )


def json_viewer(json_object, key=None):
    return _json_viewer(json_object=json_object, key=key, default=0)


json_object = [
    {"a": [1, 2, 3]},
    "two", "the", "sunny", "day", "beautiful", "mark"]
return_value = json_viewer(json_object=json_object)
st.write(return_value)
