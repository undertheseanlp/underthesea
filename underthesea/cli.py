# -*- coding: utf-8 -*-
import click
import platform
from underthesea.util import download_component
from underthesea.util.data import download_default_components


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

@main.command()
def info():
    underthesea_version = open("VERSION").read().strip()
    python_version = platform.python_version()
    system_info = f"{platform.system()}-{platform.release()}"
    print("")
    print("ENVIRONMENT")
    print(f" underthesea version : {underthesea_version}")
    print(f"      python version : {python_version}")
    print(f"  system information : {system_info}")

    print("")
    print("MODULES")
    print("       sent_tokenize : OK")
    print("       word_tokenize : OK")
    print("             pos_tag : OK")
    print("               chunk : OK")
    print("                 ner : OK")
    print("            classify : OK")
    print("           sentiment : OK")


if __name__ == "__main__":
    main()
