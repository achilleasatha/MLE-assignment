import pytest
from fastapi.testclient import TestClient

from prototype.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.integration
def test_inference_endpoint(client):
    test_data = {
        "data": [
            {"name": "Product 1", "description": "Stripe", "product_id": 1},
            {"name": "Product 2", "description": "Check", "product_id": 2},
        ]
    }

    # Make a POST request to the inference endpoint
    response = client.post("/infer", json=test_data)

    # Assert the response status code is 200
    assert response.status_code == 200

    # Assert the response body contains the expected data
    expected_response = [
        {"pattern": "Stripe", "product_id": 1},
        {"pattern": "Check", "product_id": 2},
    ]
    assert response.json() == expected_response
