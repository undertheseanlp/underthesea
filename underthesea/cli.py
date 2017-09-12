# -*- coding: utf-8 -*-
import click
try:
    from underthesea.util import download_component
except:
    from util import download_component


@click.group()
def main(args=None):
    """Console script for underthesea"""
    pass


@main.command()
@click.argument('component')
def download(component):
    download_component(component)


if __name__ == "__main__":
    main()
