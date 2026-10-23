import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app
from predictor import Predictor


class FakeBooster:
    def __init__(self, value=12.34):
        self.value = value
        self.last_input = None

    def predict(self, frame):
        self.last_input = frame
        return np.array([self.value])


@pytest.fixture()
def booster():
    return FakeBooster()


@pytest.fixture()
def client(booster):
    app.state.predictor = Predictor(booster)
    return TestClient(app)