from rest_framework.test import APITestCase
from languages.models import Article
from django.urls import reverse


class ArticleViewSetTestCase(APITestCase):
    def setUp(self):
        self.article1 = Article.objects.create(
            title="Title 1", description="Description 1", text="Text 1"
        )
        self.article2 = Article.objects.create(
            title="Title 2", description="Description 2", text="Text 2"
        )

    def test_list(self):
        url = reverse("articles")
        response = self.client.get(url)

        # Verify that the response is correct
        self.assertEqual(response.status_code, 200)
