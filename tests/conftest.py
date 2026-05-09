import pytest
from fastapi.testclient import TestClient

from financialnewssentiment.app.main import app


@pytest.fixture
def client():
    """
    To run aplication
    """
    return TestClient(app)
