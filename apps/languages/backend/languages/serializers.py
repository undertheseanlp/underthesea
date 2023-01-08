from rest_framework import serializers
from .models import Article, Collection, Task


class TaskSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Task
        fields = ("id", "content", "status")


class ArticleSerializer(serializers.HyperlinkedModelSerializer):
    collection_id = serializers.PrimaryKeyRelatedField(
        source="collection", read_only=True
    )

    class Meta:
        model = Article
        fields = ("id", "title", "description", "text", "collection_id")


class ArticleListSerializer(serializers.HyperlinkedModelSerializer):
    collection_id = serializers.PrimaryKeyRelatedField(
        source="collection", read_only=True
    )

    class Meta:
        model = Article
        fields = ("id", "title", "description", "collection_id")


class ArticleDetailSerializer(serializers.HyperlinkedModelSerializer):
    collection_id = serializers.PrimaryKeyRelatedField(
        source="collection", read_only=True
    )

    class Meta:
        model = Article
        fields = ("id", "title", "description", "text", "collection_id")


class CollectionSerializer(serializers.HyperlinkedModelSerializer):
    # articles = ArticleSerializer(many=True, read_only=True)
    class Meta:
        model = Collection
        fields = ("id", "name")
