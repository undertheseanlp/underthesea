from rest_framework import serializers
from .models import Article, Collection, Task


class TaskSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Task
        fields = ("id", "content", "status")


class ArticleSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Article
        fields = ("id", "title", "description", "text")


class CollectionSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Collection
        fields = ("id", "name")
