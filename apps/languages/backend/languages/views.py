from .models import Article, Collection, Task
from .serializers import ArticleSerializer, CollectionSerializer, TaskSerializer
from rest_framework import viewsets
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

class TaskViewSet(viewsets.ModelViewSet):
    queryset = Task.objects.all()
    serializer_class = TaskSerializer


class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer

    def list(self, request, *args, **kwargs):
        collection_id = kwargs.get("collection_id")
        if collection_id:
            collection = get_object_or_404(Collection, id=collection_id)
            articles = collection.articles.all()
        else:
            articles = self.queryset
        serializer = self.get_serializer(articles, many=True)
        return Response(serializer.data)

class CollectionViewSet(viewsets.ModelViewSet):
    queryset = Collection.objects.all()
    serializer_class = CollectionSerializer