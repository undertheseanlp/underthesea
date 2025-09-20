from django.db import models


class Task(models.Model):
    content = models.CharField(max_length=30)
    status = models.CharField(max_length=30)


class Collection(models.Model):
    name = models.TextField()


class Article(models.Model):
    title = models.TextField()
    description = models.TextField()
    text = models.TextField(default="")
    collection = models.ForeignKey(
        Collection,
        on_delete=models.CASCADE,
        default=None,
        null=True,
        related_name="articles",
    )
