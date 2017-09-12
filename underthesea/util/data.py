#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
    from underthesea.util import download_component
except:
    from util import download_component


# download default components
default_components = ["classification.fasttext.model"]
[download_component(component) for component in default_components]

