from languages.models import Article, Collection, Task
from languages.serializers import (
    ArticleDetailSerializer,
    ArticleListSerializer,
    CollectionSerializer,
    TaskSerializer,
)
from rest_framework import viewsets
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from rest_framework.pagination import LimitOffsetPagination


class TaskViewSet(viewsets.ModelViewSet):
    queryset = Task.objects.all()
    serializer_class = TaskSerializer


class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleListSerializer
    pagination_class = LimitOffsetPagination

    def list(self, request, *args, **kwargs):
        collection_id = kwargs.get("collection_id")
        if collection_id:
            collection = get_object_or_404(Collection, id=collection_id)
            articles = collection.articles.all()
        else:
            articles = self.queryset
        page = self.paginate_queryset(articles)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        serializer = self.get_serializer(articles, many=True)
        return Response(serializer.data)

    def retrieve(self, request, *args, **kwargs):
        self.serializer_class = ArticleDetailSerializer
        output = super().retrieve(request, *args, **kwargs)
        self.serializer_class = ArticleListSerializer
        return output


class CollectionViewSet(viewsets.ModelViewSet):
    queryset = Collection.objects.all()
    serializer_class = CollectionSerializer
