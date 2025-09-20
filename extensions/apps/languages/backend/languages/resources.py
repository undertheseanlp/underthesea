from import_export import resources
from .models import Article


class ArticleResource(resources.ModelResource):
    class Meta:
        model = Article

    def import_data(self, dataset, dry_run=False, **kwargs):
        for row in dataset.dict:
            person = Article(
                title=row["title"],
                description=row["description"],
                text=row["text"],
            )
            if not dry_run:
                person.save()
        return len(dataset)
