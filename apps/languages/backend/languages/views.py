from .models import Article, Collection, Task
from .serializers import ArticleSerializer, CollectionSerializer, TaskSerializer
from rest_framework import viewsets
from rest_framework.response import Response

class TaskViewSet(viewsets.ModelViewSet):
    queryset = Task.objects.all()
    serializer_class = TaskSerializer


class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer


class CollectionViewSet(viewsets.ModelViewSet):
    queryset = Collection.objects.all()
    serializer_class = CollectionSerializer

    # def list(self, request):
    #     # Get the collections
    #     collections = self.get_queryset()

    #     # Include the articles in the response
    #     serializer = self.get_serializer(collections, many=True)
    #     data = serializer.data
    #     for i, collection in enumerate(collections):
    #         articles = Article.objects.filter(collection=collection)
    #         data[i]['articles'] = ArticleSerializer(articles, many=True).data

    #     return Response(data)