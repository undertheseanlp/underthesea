from rest_framework.test import APITestCase
from languages.models import Collection
from django.urls import reverse


class CollectionTestCase(APITestCase):
    def setUp(self):
        self.article1 = Collection.objects.create(name="Name 1")
        self.article2 = Collection.objects.create(name="Name 2")
        self.url = reverse("collections-list")

    def test_list(self):
        params = {"limit": 10, "offset": 0}
        response = self.client.get(self.url, params)

        # Verify that the response is correct
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["count"], 2)
        self.assertEqual(len(data["results"]), 2)

    def test_create(self):
        item = {"name": "New Name"}
        response = self.client.post(self.url, item, format="json")

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["name"], item["name"])

    def test_retrieve(self):
        url = reverse("collections-detail", kwargs={"pk": self.article1.id})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["name"], self.article1.name)

    def test_detroy(self):
        url = reverse('collections-detail', kwargs={'pk': self.article1.id})
        response = self.client.delete(url)

        # Verify that the response is correct
        self.assertEqual(response.status_code, 204)  # The HTTP status code for a successful DELETE request
        self.assertEqual(Collection.objects.count(), 1)
