from .models import Article, Task
from.serializers import ArticleSerializer, TaskSerializer
from rest_framework import viewsets

class TaskViewSet(viewsets.ModelViewSet): 
  queryset = Task.objects.all() 
  serializer_class = TaskSerializer 

class ArticleViewSet(viewsets.ModelViewSet): 
  queryset = Article.objects.all() 
  serializer_class = ArticleSerializer 
  
