# flake8: noqa: E402
import tablib
from import_export import resources
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "languages.settings")
import django

django.setup()
from languages.models import Article, Collection
from os.path import dirname, join, abspath

collection = Collection(name="VLC")
collection.save()
collection_id = collection.pk

pwd = dirname(abspath(__file__))
corpus = join(
    dirname(dirname(dirname(pwd))),
    "datasets",
    "CP_Vietnamese-VLC",
    "output",
    "processed",
)
files = os.listdir(corpus)
ArticleResource = resources.modelresource_factory(model=Article)()
dataset = tablib.Dataset(headers=["title", "description", "text", "collection"])
for file in files:
    name = file.split(".")[0]
    title = name
    description = name
    with open(join(corpus, file)) as f:
        text = f.read().strip()
    item = (title, description, text, collection_id)
    dataset.append(item)
result = ArticleResource.import_data(dataset, dry_run=False)
print("Finish")
