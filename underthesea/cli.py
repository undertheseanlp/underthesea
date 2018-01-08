# -*- coding: utf-8 -*-
import click

try:
    from underthesea.util import download_component
    from underthesea.util.data import download_default_components
except:
    from util import download_component
    from util.data import download_default_components


@click.group()
def main(args=None):
    """Console script for underthesea"""
    pass


@main.command()
@click.argument('component')
def download(component):
    download_component(component)


@main.command()
def data():
    download_default_components()


if __name__ == "__main__":
    main()
