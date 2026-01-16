from locust import HttpUser, task, between
import random


class QuickstartUser(HttpUser):
    wait_time = between(1, 5)  # wait between 1 and 5 seconds between tasks

    @task
    def get_root(self):
        self.client.get("/")  # Simulate a GET request to the root endpoint

    @task(3)
    def post_predict(self):
        titles = ["Breaking News", "Latest Updates", "Exclusive Report", "Top Story", "In-Depth Analysis"]
        texts = [
            "This is a sample news article text.",
            "The quick brown fox jumps over the lazy dog.",
            "In today's news, we explore the impact of technology on society.",
            "Sports events are resuming with new safety protocols in place.",
            "Economic trends show a significant shift in market dynamics.",
        ]
        title = random.choice(titles)
        text = random.choice(texts)
        payload = {"title": title, "text": text}
        self.client.post("/predict", json=payload)  # Simulate a POST request to the predict endpoint
