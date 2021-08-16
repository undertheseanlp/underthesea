from github import Github
import os


def upload_datasets():
    pass


def download_datasets():
    pass


def build_datasets():
    pass


if __name__ == '__main__':
    GITHUB_ACCESS_TOKEN = os.environ["GITHUB_ACCESS_TOKEN"]
    g = Github(GITHUB_ACCESS_TOKEN)
    release = g.get_repo("undertheseanlp/underthesea").get_release("open-data")
    assets = release.get_assets("abc.txt")
    print(0)
