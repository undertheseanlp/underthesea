from os.path import join, dirname, abspath
import shutil
from underthesea.file_utils import DATASETS_FOLDER
import click
from github import Github

corpus = join(dirname(dirname(abspath(__file__))), "output", "processed")
corpus_path = join(DATASETS_FOLDER, "CP_Vietnamese_VLC_v2_2022")
zip_corpus_path = join(DATASETS_FOLDER, "CP_Vietnamese_VLC_v2_2022.zip")
asset_name = "CP_Vietnamese_VLC_v2_2022.zip"


@click.group()
def main(args=None):
    """Console script for publish"""
    pass


@main.command()
@click.argument("access_token", required=True)
def github(access_token):
    publish_local()
    publish_github(access_token)


def publish_github(access_token):
    g = Github(access_token)
    repo = g.get_repo("undertheseanlp/underthesea")
    release = repo.get_release("resources")
    release.upload_asset(zip_corpus_path, name=asset_name)
    print("Successfully uploaded zip file!")


def publish_local():
    try:
        shutil.rmtree(corpus_path)
    except Exception as e:
        print(e)
    finally:
        shutil.copytree(corpus, corpus_path)

    shutil.make_archive(corpus_path, "zip", corpus_path)


@main.command()
def local():
    publish_local()


if __name__ == "__main__":
    main()
