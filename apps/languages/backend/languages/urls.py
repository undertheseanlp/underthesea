"""languages URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from rest_framework import routers
from .views import ArticleViewSet, TaskViewSet, CollectionViewSet

router = routers.DefaultRouter()
router.register(r'tasks', TaskViewSet)
router.register(r'collections', CollectionViewSet, basename="collections")
router.register(r'articles', ArticleViewSet, basename="articles")

urlpatterns = [
    path('admin/', admin.site.urls),
    path(r'api/', include(router.urls)),
    path(r'api/collections/<int:collection_id>/articles/', ArticleViewSet.as_view({"get": "list"}), name="list_articles")
]
