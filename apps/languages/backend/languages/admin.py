from django.contrib import admin
from import_export.admin import ImportExportModelAdmin
from .models import Article


@admin.register(Article)
class ArticleAdmin(ImportExportModelAdmin):
    list_display = ["title", "description", "text"]
