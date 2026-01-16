from fastapi.testclient import TestClient
from mlops_project.api import app

client = TestClient(app)


# Test the root endpoint, code credited to Nicki Skafte Detlefsen and his MLOps course at 02476 - DTU.
# It has been copied from the exercises provided in the course.
def test_read_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to the News Classification API"}


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
