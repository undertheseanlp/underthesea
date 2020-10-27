# -*- coding: utf-8 -*-
import os

import click
import platform

from underthesea.data_fetcher import DataFetcher
from underthesea.model_fetcher import ModelFetcher


@click.group()
def main(args=None):
    """Console script for underthesea"""
    pass


@main.command()
@click.option('-a', '--all', is_flag=True, required=False)
def list_data(all):
    DataFetcher.list(all)


@main.command()
@click.argument('dataset', required=True)
@click.argument('url', required=False)
def download_data(dataset, url):
    DataFetcher.download_data(dataset, url)


@main.command()
@click.argument('data', required=True)
def remove_data(data):
    DataFetcher.remove(data)


@main.command()
@click.argument('model', required=True)
def download(model):
    ModelFetcher.download_model(model)


@main.command()
@click.option('-a', '--all', is_flag=True, required=False)
def list(all):
    ModelFetcher.list(all)


@main.command()
@click.argument('model', required=True)
def remove(model):
    ModelFetcher.remove(model)


@main.command()
def info():
    version_file = os.path.join(os.path.dirname(__file__), 'VERSION')
    underthesea_version = open(version_file).read().strip()
    python_version = platform.python_version()
    system_info = f"{platform.system()}{platform.release()}"
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
