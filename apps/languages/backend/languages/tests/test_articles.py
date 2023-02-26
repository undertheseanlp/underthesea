from rest_framework.test import APITestCase
from languages.models import Article
from django.urls import reverse


class ArticlesTestCase(APITestCase):
    def setUp(self):
        self.article1 = Article.objects.create(
            title="Title 1", description="Description 1", text="Text 1"
        )
        self.article2 = Article.objects.create(
            title="Title 2", description="Description 2", text="Text 2"
        )

    def test_list(self):
        url = reverse("articles-list")
        params = {"limit": 10, "offset": 0}
        response = self.client.get(url, params)

        # Verify that the response is correct
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["count"], 2)
        self.assertEqual(len(data["results"]), 2)

    def test_create(self):
        url = reverse("articles-list")
        item = {"title": "New Article", "description": "New Description"}
        response = self.client.post(url, item, format="json")

        # Verify that the response is correct
        self.assertEqual(
            response.status_code, 201
        )  # The HTTP status code for a successful POST request
        self.assertEqual(response.data["title"], item["title"])
        self.assertEqual(response.data["description"], item["description"])

    def test_retrieve(self):
        url = reverse("articles-detail", kwargs={"pk": self.article1.id})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["id"], self.article1.id)
        self.assertEqual(response.data["title"], self.article1.title)

    def test_detroy(self):
        url = reverse("articles-detail", kwargs={"pk": self.article1.id})
        response = self.client.delete(url)

        # Verify that the response is correct
        self.assertEqual(
            response.status_code, 204
        )  # The HTTP status code for a successful DELETE request
        self.assertEqual(Article.objects.count(), 1)
