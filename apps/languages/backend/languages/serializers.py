from rest_framework import serializers
from .models import Article, Collection, Task


class TaskSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Task
        fields = ("id", "content", "status")


class ArticleSerializer(serializers.HyperlinkedModelSerializer):
    # collection = serializers.HyperlinkedRelatedField(view_name='collection-detail', queryset=Collection.objects.all())
    class Meta:
        model = Article
        fields = ("id", "title", "description", "text", "collection")


class CollectionSerializer(serializers.HyperlinkedModelSerializer):
    articles = ArticleSerializer(many=True, read_only=True)
    class Meta:
        model = Collection
        fields = ("id", "name", "articles")
