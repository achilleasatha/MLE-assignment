from locust import HttpUser, between, task


class MyUser(HttpUser):
    host = "http://localhost:8000"

    wait_time = between(1, 2)

    @task
    def infer(self):
        products = {
            "data": [
                {
                    "name": "Test Product 1",
                    "description": "Description 1",
                    "product_id": 1,
                },
                {
                    "name": "Test Product 2",
                    "description": "Description 2",
                    "product_id": 2,
                },
            ]
        }
        self.client.post("/infer", json=products)

    def on_start(self):
        pass
