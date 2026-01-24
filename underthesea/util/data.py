#!/usr/bin/env python
from underthesea.util import download_component


def download_default_components():
    # download default components
    default_components = ["classification.vntc.model"]
    [download_component(component) for component in default_components]


if __name__ == '__main__':
    download_default_components()
