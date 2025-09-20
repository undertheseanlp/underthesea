"""service URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
# flake8: noqa
from django.contrib import admin
from django.urls import path
from django.contrib.staticfiles.urls import staticfiles_urlpatterns


admin.autodiscover()
from service import views

urlpatterns = [
    path('', views.index, name='index'),
    path('word_sent', views.word_sent),
    path('pos_tag', views.pos_tag),
    path('chunking', views.chunking),
    path('ner', views.ner),
    path('classification', views.classification),
    path('sentiment', views.sentiment),
    path('dictionary', views.dictionary),
    path('ipa', views.ipa),
    
    path('admin/', admin.site.urls),
]

urlpatterns += staticfiles_urlpatterns()
