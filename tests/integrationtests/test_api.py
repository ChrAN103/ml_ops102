from pathlib import Path
from fastapi.testclient import TestClient
import pytest
from mlops_project.api import app
import os

client = TestClient(app)
MODEL_PATH = Path("models/model.pt")


# Test the root endpoint, code credited to Nicki Skafte Detlefsen and his MLOps course at 02476 - DTU.
# It has been copied from the exercises provided in the course.
@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model file does not exist")
def test_read_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["message"] == "OK"
        assert json_response["status-code"] == 200
        assert "device" in json_response


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model file does not exist")
def test_predict():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"title": "Breaking News", "text": "This is a sample news article text."},
        )
        assert response.status_code == 200
        json_response = response.json()
        assert "prediction" in json_response
        assert "prob" in json_response


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model file does not exist")
@pytest.mark.skipif(os.getenv("GOOGLE_APPLICATION_CREDENTIALS") is None, reason="GCS credentials not configured")
def test_data_drift_report_integration():
    """Integration test for data drift report with real GCS access."""
    with TestClient(app) as client:
        response = client.get("/report?n=10")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Evidently" in response.text or "Data Drift" in response.text
