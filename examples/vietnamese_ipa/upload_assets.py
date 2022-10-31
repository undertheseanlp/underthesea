import os
from os import listdir
from os.path import join

from github import Github

from underthesea.utils import logger

access_token = os.environ['GITHUB_ACCESS_TOKEN']
g = Github(access_token)
repo = g.get_repo('undertheseanlp/underthesea')
release = repo.get_release('open-data-voice-ipa')
assets = {}
i = 0
for asset in release.get_assets():
    i += 1
    if i % 200 == 0:
        logger.info(f"Load {i} items")
    name = asset.name
    assets[name] = asset

SOUNDS_FOLDER = join("../text_normalize/outputs", "sound", "zalo")
names = listdir(SOUNDS_FOLDER)


def generate_id(word):
    return "".join([hex(ord(c)) for c in word])


for name in names:
    path = join(SOUNDS_FOLDER, name)
    word = name.split(".")[0]
    word_id = generate_id(word)
    asset_name = word_id + ".mp3"
    if asset_name in assets:
        # asset = assets[asset_name]
        # asset.delete_asset()
        continue
    logger.info(f'Upload file {asset_name} for word {name}')
    release.upload_asset(path, name=asset_name)
