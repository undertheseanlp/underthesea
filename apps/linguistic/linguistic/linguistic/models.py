from django.db import models

class Task(models.Model):
    content = models.CharField(max_length=30)
    status = models.CharField(max_length=30)