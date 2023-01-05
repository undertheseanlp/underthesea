import tablib
from import_export import resources
from models import Article


book_resource = resources.modelresource_factory(model=Article)()
dataset = tablib.Dataset(['', 'New book'], headers=['title', 'description', 'text'])
result = book_resource.import_data(dataset, dry_run=True)
print(result.has_errors())
# # Import data from a CSV file
# import_person_data('sample_data.csv')