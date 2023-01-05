import tablib
from import_export import resources
import django
django.setup()
from languages.models import Article

resource = resources.modelresource_factory(model=Article)()
dataset = tablib.Dataset(headers=['title', 'description', 'text'])
for item in [('a', 'b', 'c'), ('d', 'e', 'f')]:
    dataset.append(item)
result = resource.import_data(dataset, dry_run=False)
print('Finish')