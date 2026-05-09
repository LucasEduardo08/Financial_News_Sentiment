from http import HTTPStatus

from financialnewssentiment.app.model_loader import load_model
from financialnewssentiment.app.preprocess_text import preprocess_text


def test_health(client):
    response = client.get("/health")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"status": "Ok"}


def test_model_info(client):
    response = client.get("/model_info")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"model_type": "Keras","version": "0.1.0"}


def test_predict(client):
    response = client.post(
        "/predict",
        json={
            "text": "The company's shares have climbed 11% since the beginning of the year. In the final minutes of trading on Monday, shares hit $13.16, a climb of 12% in the last 12 months."
        }
    )

    assert response.status_code == HTTPStatus.OK
