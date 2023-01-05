import tablib
from import_export import resources
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'languages.settings')
import django
django.setup()
from languages.models import Article, Collection


collection = Collection(name='VLC')
collection.save()
collection_id = collection.pk

ArticleResource = resources.modelresource_factory(model=Article)()
dataset = tablib.Dataset(headers=['title', 'description', 'text', 'collection'])
for item in [('a', 'b', 'c', collection_id), ('d', 'e', 'f', collection_id)]:
    dataset.append(item)
result = ArticleResource.import_data(dataset, dry_run=False)
print('Finish')